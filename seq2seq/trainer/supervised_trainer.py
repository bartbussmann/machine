from __future__ import division
import logging
import os
import random
import time
import datetime
import shutil

import torch
import torchtext
from torch import optim
from tensorboardX import SummaryWriter
from torch.autograd import Variable

import seq2seq
from seq2seq.evaluator import Evaluator
from seq2seq.loss import NLLLoss, Variance
from seq2seq.optim import Optimizer
from seq2seq.util.checkpoint import Checkpoint

class SupervisedTrainer(object):
    """ The SupervisedTrainer class helps in setting up a training framework in a
    supervised setting.

    Args:
        expt_dir (optional, str): experiment Directory to store details of the experiment,
            by default it makes a folder in the current directory to store the details (default: `experiment`).
        loss (seq2seq.loss.loss.Loss, optional): loss for training, (default: seq2seq.loss.NLLLoss)
        batch_size (int, optional): batch size for experiment, (default: 64)
        checkpoint_every (int, optional): number of epochs to checkpoint after, (default: 100)
    """
    def __init__(self, expt_dir='experiment', loss=NLLLoss(), batch_size=64,
                 random_seed=None,
                 checkpoint_every=100, print_every=100):
        self._trainer = "Simple Trainer"
        self.random_seed = random_seed
        if random_seed is not None:
            random.seed(random_seed)
            torch.manual_seed(random_seed)
        self.loss = loss
        self.evaluator = Evaluator(loss=self.loss, batch_size=batch_size)
        self.optimizer = None
        self.checkpoint_every = checkpoint_every
        self.print_every = print_every

        if not os.path.isabs(expt_dir):
            expt_dir = os.path.join(os.getcwd(), expt_dir)
        self.expt_dir = expt_dir
        if not os.path.exists(self.expt_dir):
            os.makedirs(self.expt_dir)
        self.batch_size = batch_size

        self.logger = logging.getLogger(__name__)

        if torch.cuda.is_available():
            self.tensorboard_dir = os.path.join("tensorboard_runs", datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S") + '-cuda' + str(torch.cuda.current_device()))
        else:
            self.tensorboard_dir = os.path.join("tensorboard_runs", datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

        self.writer = SummaryWriter(self.tensorboard_dir)

    def _train_batch(self, input_variable, input_lengths, target_variable, model, teacher_forcing_ratio, reg_scale, run_step):
        loss = self.loss
        # Forward propagation
        decoder_outputs, decoder_hidden, other = model(input_variable, input_lengths, target_variable,
                                                       teacher_forcing_ratio=teacher_forcing_ratio)
        # Get loss
        loss.reset()
        for step, step_output in enumerate(decoder_outputs):
            batch_size = target_variable.size(0)
            loss.eval_batch(step_output.contiguous().view(batch_size, -1), target_variable[:, step + 1])

        # add regularization loss
        input_vocab_size = model.encoder.vocab_size
        output_vocab_size = model.decoder.vocab_size
        variance, _ = Variance.get_variance(input_variable, other['sequence'], other['attention_score'], input_vocab_size, output_vocab_size, reg_scale)  

        self.writer.add_scalar("variance/train", variance, run_step)

        regularizaton = -reg_scale * variance
        loss.acc_loss += regularizaton 

        # Backward propagation
        model.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.get_loss()

    def _train_epoches(self, data, model, n_epochs, start_epoch, start_step,
                       dev_data=None, teacher_forcing_ratio=0, top_k=5, reg_scale=1000):
        log = self.logger

        print_loss_total = 0  # Reset every print_every
        epoch_loss_total = 0  # Reset every epoch

        device = None if torch.cuda.is_available() else -1
        batch_iterator = torchtext.data.BucketIterator(
            dataset=data, batch_size=self.batch_size,
            sort=False, sort_within_batch=True,
            sort_key=lambda x: len(x.src),
            device=device, repeat=False)

        steps_per_epoch = len(batch_iterator)
        total_steps = steps_per_epoch * n_epochs

        step = start_step
        step_elapsed = 0

        # store initial model to be sure at least one model is stored
        eval_data = dev_data or data
        loss, accuracy, seq_accuracy, variance = self.evaluator.evaluate(model, eval_data, reg_scale, self.writer, step)
        loss_best = top_k*[loss]
        var_best = top_k*[variance]
        best_checkpoints = top_k*[None]
        model_name = 'var_%.2f_acc_%.2f_seq_acc_%.2f_ppl_%.2f_s%d' % (variance, accuracy, seq_accuracy, loss, 0)
        best_checkpoints[0] = model_name

        self.writer.add_scalar("loss/validation", loss, step)
        self.writer.add_scalar("variance/validation", variance, step)

        Checkpoint(model=model,
                   optimizer=self.optimizer,
                   epoch=start_epoch, step=start_step,
                   input_vocab=data.fields[seq2seq.src_field_name].vocab,
                   output_vocab=data.fields[seq2seq.tgt_field_name].vocab).save(self.expt_dir, name=model_name)


        for epoch in range(start_epoch, n_epochs + 1):
            log.debug("Epoch: %d, Step: %d" % (epoch, step))

            batch_generator = batch_iterator.__iter__()

            # consuming seen batches from previous training
            for _ in range((epoch - 1) * steps_per_epoch, step):
                next(batch_generator)

            model.train(True)
            for batch in batch_generator:
                step += 1
                step_elapsed += 1

                input_variables, input_lengths = getattr(batch, seq2seq.src_field_name)
                target_variables = getattr(batch, seq2seq.tgt_field_name)

                loss = self._train_batch(input_variables, input_lengths.tolist(), target_variables, model, teacher_forcing_ratio, reg_scale, step)

                self.writer.add_scalar("loss/train", loss, step)

                # Record average loss
                print_loss_total += loss
                epoch_loss_total += loss

                # print log info according to print_every parm
                if step % self.print_every == 0 and step_elapsed > self.print_every:
                    print_loss_avg = print_loss_total / self.print_every
                    print_loss_total = 0
                    log_msg = 'Progress: %d%%, Train %s: %.4f' % (
                        step / total_steps * 100,
                        self.loss.name,
                        print_loss_avg)
                    log.info(log_msg)

                # check if new model should be saved
                if step % self.checkpoint_every == 0 or step == total_steps:
                    # compute dev loss
                    loss, accuracy, seq_accuracy, variance = self.evaluator.evaluate(model, eval_data, reg_scale, self.writer, step)
                    max_eval_loss = max(loss_best)
                    max_variance = max(var_best)

                    self.writer.add_scalar("loss/validation", loss, step)
                    self.writer.add_scalar("variance/validation", variance, step)
                    
                    if loss < max_eval_loss:
                    # if variance > max_variance:
                            index_max = loss_best.index(max_eval_loss)
                            # index_max = var_best.index(max_variance)
                            # rm prev model
                            if best_checkpoints[index_max] is not None:
                                shutil.rmtree(os.path.join(self.expt_dir, best_checkpoints[index_max]))
                            model_name = 'var_%.2f_acc_%.2f_seq_acc_%.2f_ppl_%.2f_s%d' % (variance, accuracy, seq_accuracy, loss, step)

                            self.logger.debug("Saved checkpoint {}".format(model_name))

                            best_checkpoints[index_max] = model_name
                            loss_best[index_max] = loss
                            var_best[index_max] = variance

                            # save model
                            Checkpoint(model=model,
                                       optimizer=self.optimizer,
                                       epoch=epoch, step=step,
                                       input_vocab=data.fields[seq2seq.src_field_name].vocab,
                                       output_vocab=data.fields[seq2seq.tgt_field_name].vocab).save(self.expt_dir, name=model_name)

            if step_elapsed == 0: continue

            epoch_loss_avg = epoch_loss_total / min(steps_per_epoch, step - start_step)
            epoch_loss_total = 0
            log_msg = "Finished epoch %d: Train %s: %.4f" % (epoch, self.loss.name, epoch_loss_avg)
            if dev_data is not None:
                dev_loss, accuracy, seq_accuracy, variance = self.evaluator.evaluate(model, dev_data, reg_scale, self.writer, step)
                self.optimizer.update(dev_loss, epoch)
                log_msg += ", Dev %s: %.4f, Accuracy: %.4f, Sequence Accuracy: %.4f, Variance: %.4f" % (self.loss.name, dev_loss, accuracy, seq_accuracy, variance)
                model.train(mode=True)
            else:
                self.optimizer.update(epoch_loss_avg, epoch)

            log.info(log_msg)

    def train(self, model, data, num_epochs=5,
              resume=False, dev_data=None,
              optimizer=None, teacher_forcing_ratio=0,
              learning_rate=0.001, checkpoint_path=None, top_k=5, reg_scale=1000):
        """ Run training for a given model.

        Args:
            model (seq2seq.models): model to run training on, if `resume=True`, it would be
               overwritten by the model loaded from the latest checkpoint.
            data (seq2seq.dataset.dataset.Dataset): dataset object to train on
            num_epochs (int, optional): number of epochs to run (default 5)
            resume(bool, optional): resume training with the latest checkpoint, (default False)
            dev_data (seq2seq.dataset.dataset.Dataset, optional): dev Dataset (default None)
            optimizer (seq2seq.optim.Optimizer, optional): optimizer for training
               (default: Optimizer(pytorch.optim.Adam, max_grad_norm=5))
            teacher_forcing_ratio (float, optional): teaching forcing ratio (default 0)
            learing_rate (float, optional): learning rate used by the optimizer (default 0.001)
            checkpoint_path (str, optional): path to load checkpoint from in case training should be resumed
            top_k (int): how many models should be stored during training
            reg_scale (float): how to scale the regularization term wrt the other loss
        Returns:
            model (seq2seq.models): trained model.
        """
        # If training is set to resume
        if resume:
            resume_checkpoint = Checkpoint.load(checkpoint_path)
            model = resume_checkpoint.model
            self.optimizer = resume_checkpoint.optimizer

            # A walk around to set optimizing parameters properly
            resume_optim = self.optimizer.optimizer
            defaults = resume_optim.param_groups[0]
            defaults.pop('params', None)
            defaults.pop('initial_lr', None)
            self.optimizer.optimizer = resume_optim.__class__(model.parameters(), **defaults)

            start_epoch = resume_checkpoint.epoch
            step = resume_checkpoint.step
        else:
            start_epoch = 1
            step = 0

            def get_optim(optim_name):
                optims = {'adam': optim.Adam, 'adagrad': optim.Adagrad,
                          'adadelta': optim.Adadelta, 'adamax': optim.Adamax,
                          'rmsprop': optim.RMSprop, 'sgd': optim.SGD,
                           None:optim.Adam}
                return optims[optim_name]

            self.optimizer = Optimizer(get_optim(optimizer)(model.parameters(), lr=learning_rate),
                                       max_grad_norm=5)

        self.logger.info("Optimizer: %s, Scheduler: %s" % (self.optimizer.optimizer, self.optimizer.scheduler))

        self._train_epoches(data, model, num_epochs,
                            start_epoch, step, dev_data=dev_data,
                            teacher_forcing_ratio=teacher_forcing_ratio,
                            top_k=top_k, reg_scale=reg_scale)

        # export scalar data to JSON for external processing
        self.writer.export_scalars_to_json(os.path.join(self.tensorboard_dir, "all_scalars.json"))
        self.writer.close()

        return model
