import tensorflow as tf
import numpy as np
import os
import datetime
import time

from model import AttLSTM
import data_helpers
from configure import FLAGS


def validation():
    np.random.seed(10)
    mask = np.ones(shape=[FLAGS.sequence_length]).nonzero()

    for fold in range(1, FLAGS.k_fold+1):
        print("Attribute&Value Fold {}: ".format(fold))
        with tf.device('/cpu:0'):
            x_text_tra, x_position_tra, y_tra, x_text_val, x_position_val, y_val = \
                data_helpers.load_k_fold_data(FLAGS.av_data_path, fold)

        print("Train/Validation set size: {}/{}\n".format(len(y_tra), len(y_val)))

        with tf.Graph().as_default():
            session_conf = tf.ConfigProto(
                allow_soft_placement=FLAGS.allow_soft_placement,
                log_device_placement=FLAGS.log_device_placement)
            session_conf.gpu_options.allow_growth = FLAGS.gpu_allow_growth
            sess = tf.Session(config=session_conf)
            with sess.as_default():
                model = AttLSTM(
                    sequence_length=FLAGS.sequence_length,
                    num_classes=2,
                    vocab_size=FLAGS.vocab_size,
                    embedding_size=FLAGS.embedding_dim,
                    hidden_size=FLAGS.hidden_size,
                    l2_reg_lambda=FLAGS.l2_reg_lambda)

                # Define Training procedure
                global_step = tf.Variable(0, name="global_step", trainable=False)
                optimizer = tf.train.AdadeltaOptimizer(FLAGS.learning_rate, FLAGS.decay_rate, 1e-6)
                gvs = optimizer.compute_gradients(model.loss)
                capped_gvs = [(tf.clip_by_value(grad, -1.0, 1.0), var) for grad, var in gvs]
                train_op = optimizer.apply_gradients(capped_gvs, global_step=global_step)

                # Output directory for models and summaries
                timestamp = str(int(time.time()))
                out_dir = os.path.abspath(os.path.join(os.path.curdir, "runs", timestamp))
                print("Writing to {}\n".format(out_dir))

                # Summaries for loss and accuracy
                loss_summary = tf.summary.scalar("loss", model.loss)
                acc_summary = tf.summary.scalar("accuracy", model.accuracy)

                # Train Summaries
                train_summary_op = tf.summary.merge([loss_summary, acc_summary])
                train_summary_dir = os.path.join(out_dir, "summaries", "train")
                train_summary_writer = tf.summary.FileWriter(train_summary_dir, sess.graph)

                # Validation summaries
                dev_summary_op = tf.summary.merge([loss_summary, acc_summary])
                dev_summary_dir = os.path.join(out_dir, "summaries", "val")
                dev_summary_writer = tf.summary.FileWriter(dev_summary_dir, sess.graph)

                # Checkpoint directory. Tensorflow assumes this directory already exists so we need to create it
                checkpoint_dir = os.path.abspath(os.path.join(out_dir, "checkpoints"))
                checkpoint_prefix = os.path.join(checkpoint_dir, "model")
                if not os.path.exists(checkpoint_dir):
                    os.makedirs(checkpoint_dir)
                saver = tf.train.Saver(tf.global_variables(), max_to_keep=FLAGS.num_checkpoints)

                # Initialize all variables
                sess.run(tf.global_variables_initializer())

                # Pre-trained word2vec
                pretrain_W = np.load("embedding.npy")
                norm = max(abs(pretrain_W.max()), abs(pretrain_W.min()))
                pretrain_W /= 4 * norm
                sess.run(model.W_text.assign(pretrain_W))
                print("Success to load pre-trained word2vec model!\n")

                # Generate batches
                batches = data_helpers.batch_iter(x_text_tra, x_position_tra, y_tra, FLAGS.batch_size, FLAGS.num_epochs)
                # Training loop. For each batch...
                best_accuracy = 0.0  # For save checkpoint(model)
                for batch in batches:
                    x_text_batch, x_position_batch, y_batch = batch
                    # Train
                    feed_dict = {
                        model.input_text: x_text_batch,
                        model.input_position: x_position_batch,
                        model.input_y: y_batch,
                        model.mask: mask,
                        model.emb_dropout_keep_prob: FLAGS.emb_dropout_keep_prob,
                        model.rnn_dropout_keep_prob: FLAGS.rnn_dropout_keep_prob,
                        model.dropout_keep_prob: FLAGS.dropout_keep_prob
                    }
                    _, step, summaries, loss, accuracy = sess.run(
                        [train_op, global_step, train_summary_op, model.loss, model.accuracy], feed_dict)
                    train_summary_writer.add_summary(summaries, step)

                    # Training log display
                    if step % FLAGS.display_every == 0:
                        time_str = datetime.datetime.now().isoformat()
                        print("{}: step {}, loss {:g}, acc {:g}".format(time_str, step, loss, accuracy))

                    # Evaluation
                    if step % FLAGS.evaluate_every == 0:
                        print("\nEvaluation:")
                        accs = []
                        losses = []
                        val_batches = data_helpers.batch_iter(x_text_val, x_position_val, y_val, FLAGS.batch_size, 1)
                        for val_batch in val_batches:
                            x_text_val_batch, x_position_val_batch, y_val_batch = val_batch
                            feed_dict = {
                                model.input_text: x_text_val_batch,
                                model.input_position: x_position_val_batch,
                                model.input_y: y_val_batch,
                                model.mask: mask,
                                model.emb_dropout_keep_prob: 1.0,
                                model.rnn_dropout_keep_prob: 1.0,
                                model.dropout_keep_prob: 1.0
                            }
                            summaries, loss, accuracy, predictions = sess.run(
                                [dev_summary_op, model.loss, model.accuracy, model.predictions], feed_dict)
                            accs.append(accuracy)
                            losses.append(loss)
                            dev_summary_writer.add_summary(summaries, step)

                        time_str = datetime.datetime.now().isoformat()
                        acc = np.mean(accs)
                        los = np.mean(losses)
                        print("{}: step {}, loss {:g}, acc {:g}".format(time_str, step, los, acc))

                        # Model checkpoint
                        if best_accuracy < acc:
                            best_accuracy = acc
                            path = saver.save(sess, checkpoint_prefix + "-{:g}".format(best_accuracy), global_step=step)
                            print("Saved model checkpoint to {}\n".format(path))

    for fold in range(1, FLAGS.k_fold + 1):
        print("Time&Attribute Fold {}: ".format(fold))
        with tf.device('/cpu:0'):
            x_text_tra, x_position_tra, y_tra, x_text_val, x_position_val, y_val = \
                data_helpers.load_k_fold_data(FLAGS.ta_data_path, fold)

        print("Train/Validation set size: {}/{}\n".format(len(y_tra), len(y_val)))

        with tf.Graph().as_default():
            session_conf = tf.ConfigProto(
                allow_soft_placement=FLAGS.allow_soft_placement,
                log_device_placement=FLAGS.log_device_placement)
            session_conf.gpu_options.allow_growth = FLAGS.gpu_allow_growth
            sess = tf.Session(config=session_conf)
            with sess.as_default():
                model = AttLSTM(
                    sequence_length=FLAGS.sequence_length,
                    num_classes=2,
                    vocab_size=FLAGS.vocab_size,
                    embedding_size=FLAGS.embedding_dim,
                    hidden_size=FLAGS.hidden_size,
                    l2_reg_lambda=FLAGS.l2_reg_lambda)

                # Define Training procedure
                global_step = tf.Variable(0, name="global_step", trainable=False)
                optimizer = tf.train.AdadeltaOptimizer(FLAGS.learning_rate, FLAGS.decay_rate, 1e-6)
                gvs = optimizer.compute_gradients(model.loss)
                capped_gvs = [(tf.clip_by_value(grad, -1.0, 1.0), var) for grad, var in gvs]
                train_op = optimizer.apply_gradients(capped_gvs, global_step=global_step)

                # Output directory for models and summaries
                timestamp = str(int(time.time()))
                out_dir = os.path.abspath(os.path.join(os.path.curdir, "runs", timestamp))
                print("Writing to {}\n".format(out_dir))

                # Summaries for loss and accuracy
                loss_summary = tf.summary.scalar("loss", model.loss)
                acc_summary = tf.summary.scalar("accuracy", model.accuracy)

                # Train Summaries
                train_summary_op = tf.summary.merge([loss_summary, acc_summary])
                train_summary_dir = os.path.join(out_dir, "summaries", "train")
                train_summary_writer = tf.summary.FileWriter(train_summary_dir, sess.graph)

                # Validation summaries
                dev_summary_op = tf.summary.merge([loss_summary, acc_summary])
                dev_summary_dir = os.path.join(out_dir, "summaries", "val")
                dev_summary_writer = tf.summary.FileWriter(dev_summary_dir, sess.graph)

                # Checkpoint directory. Tensorflow assumes this directory already exists so we need to create it
                checkpoint_dir = os.path.abspath(os.path.join(out_dir, "checkpoints"))
                checkpoint_prefix = os.path.join(checkpoint_dir, "model")
                if not os.path.exists(checkpoint_dir):
                    os.makedirs(checkpoint_dir)
                saver = tf.train.Saver(tf.global_variables(), max_to_keep=FLAGS.num_checkpoints)

                # Initialize all variables
                sess.run(tf.global_variables_initializer())

                # Pre-trained word2vec
                # pretrain_W = np.load("embedding.npy")
                # sess.run(model.W_text.assign(pretrain_W))
                # print("Success to load pre-trained word2vec model!\n")

                # Generate batches
                batches = data_helpers.batch_iter(x_text_tra, x_position_tra, y_tra, FLAGS.batch_size, FLAGS.num_epochs)
                # Training loop. For each batch...
                best_accuracy = 0.0  # For save checkpoint(model)
                for batch in batches:
                    x_text_batch, x_position_batch, y_batch = batch
                    # Train
                    feed_dict = {
                        model.input_text: x_text_batch,
                        model.input_position: x_position_batch,
                        model.input_y: y_batch,
                        model.mask: mask,
                        model.emb_dropout_keep_prob: FLAGS.emb_dropout_keep_prob,
                        model.rnn_dropout_keep_prob: FLAGS.rnn_dropout_keep_prob,
                        model.dropout_keep_prob: FLAGS.dropout_keep_prob
                    }
                    _, step, summaries, loss, accuracy = sess.run(
                        [train_op, global_step, train_summary_op, model.loss, model.accuracy], feed_dict)
                    train_summary_writer.add_summary(summaries, step)

                    # Training log display
                    if step % FLAGS.display_every == 0:
                        time_str = datetime.datetime.now().isoformat()
                        print("{}: step {}, loss {:g}, acc {:g}".format(time_str, step, loss, accuracy))

                    # Evaluation
                    if step % FLAGS.evaluate_every == 0:
                        print("\nEvaluation:")
                        accs = []
                        losses = []
                        val_batches = data_helpers.batch_iter(x_text_val, x_position_val, y_val, FLAGS.batch_size, 1)
                        for val_batch in val_batches:
                            x_text_val_batch, x_position_val_batch, y_val_batch = val_batch
                            feed_dict = {
                                model.input_text: x_text_val_batch,
                                model.input_position: x_position_val_batch,
                                model.input_y: y_val_batch,
                                model.mask: mask,
                                model.emb_dropout_keep_prob: 1.0,
                                model.rnn_dropout_keep_prob: 1.0,
                                model.dropout_keep_prob: 1.0
                            }
                            summaries, loss, accuracy, predictions = sess.run(
                                [dev_summary_op, model.loss, model.accuracy, model.predictions], feed_dict)
                            accs.append(accuracy)
                            losses.append(loss)
                            dev_summary_writer.add_summary(summaries, step)

                        time_str = datetime.datetime.now().isoformat()
                        acc = np.mean(accs)
                        los = np.mean(losses)
                        print("{}: step {}, loss {:g}, acc {:g}".format(time_str, step, los, acc))

                        # Model checkpoint
                        if best_accuracy < acc:
                            best_accuracy = acc
                            path = saver.save(sess, checkpoint_prefix + "-{:g}".format(best_accuracy), global_step=step)
                            print("Saved model checkpoint to {}\n".format(path))


def train():
    np.random.seed(10)
    mask = np.ones(shape=[FLAGS.sequence_length]).nonzero()

    with tf.device('/cpu:0'):
        x_text, x_position, y = data_helpers.load_train_data(FLAGS.av_data_path + 'train_av.npy')

    print("Train set size: {}\n".format(len(y)))

    with tf.Graph().as_default():
        session_conf = tf.ConfigProto(
            allow_soft_placement=FLAGS.allow_soft_placement,
            log_device_placement=FLAGS.log_device_placement)
        session_conf.gpu_options.allow_growth = FLAGS.gpu_allow_growth
        sess = tf.Session(config=session_conf)
        with sess.as_default():
            model = AttLSTM(
                sequence_length=FLAGS.sequence_length,
                num_classes=2,
                vocab_size=FLAGS.vocab_size,
                embedding_size=FLAGS.embedding_dim,
                hidden_size=FLAGS.hidden_size,
                l2_reg_lambda=FLAGS.l2_reg_lambda)

            # Define Training procedure
            global_step = tf.Variable(0, name="global_step", trainable=False)
            optimizer = tf.train.AdadeltaOptimizer(FLAGS.learning_rate, FLAGS.decay_rate, 1e-6)
            gvs = optimizer.compute_gradients(model.loss)
            capped_gvs = [(tf.clip_by_value(grad, -1.0, 1.0), var) for grad, var in gvs]
            train_op = optimizer.apply_gradients(capped_gvs, global_step=global_step)

            # Output directory for models and summaries
            timestamp = str(int(time.time()))
            out_dir = os.path.abspath(os.path.join(os.path.curdir, "runs", timestamp))
            print("Writing to {}\n".format(out_dir))

            # Summaries for loss and accuracy
            loss_summary = tf.summary.scalar("loss", model.loss)
            acc_summary = tf.summary.scalar("accuracy", model.accuracy)

            # Train Summaries
            train_summary_op = tf.summary.merge([loss_summary, acc_summary])
            train_summary_dir = os.path.join(out_dir, "summaries", "train")
            train_summary_writer = tf.summary.FileWriter(train_summary_dir, sess.graph)

            # Validation summaries
            dev_summary_op = tf.summary.merge([loss_summary, acc_summary])
            dev_summary_dir = os.path.join(out_dir, "summaries", "val")
            dev_summary_writer = tf.summary.FileWriter(dev_summary_dir, sess.graph)

            # Checkpoint directory. Tensorflow assumes this directory already exists so we need to create it
            checkpoint_dir = os.path.abspath(os.path.join(out_dir, "checkpoints"))
            checkpoint_prefix = os.path.join(checkpoint_dir, "model_av")
            if not os.path.exists(checkpoint_dir):
                os.makedirs(checkpoint_dir)
            saver = tf.train.Saver(tf.global_variables(), max_to_keep=FLAGS.num_checkpoints)

            # Initialize all variables
            sess.run(tf.global_variables_initializer())

            # Pre-trained word2vec
            pretrain_W = np.load("embedding.npy")
            norm = max(abs(pretrain_W.max()), abs(pretrain_W.min()))
            pretrain_W /= 4 * norm
            sess.run(model.W_text.assign(pretrain_W))
            print("Success to load pre-trained word2vec model!\n")

            # Generate batches
            batches = data_helpers.batch_iter(x_text, x_position, y, FLAGS.batch_size, FLAGS.final_epochs_av)
            # Training loop. For each batch...
            best_accuracy = 0.0  # For save checkpoint(model)
            accs = []
            losses = []
            for batch in batches:
                x_text_batch, x_position_batch, y_batch = batch
                # Train
                feed_dict = {
                    model.input_text: x_text_batch,
                    model.input_position: x_position_batch,
                    model.input_y: y_batch,
                    model.mask: mask,
                    model.emb_dropout_keep_prob: FLAGS.emb_dropout_keep_prob,
                    model.rnn_dropout_keep_prob: FLAGS.rnn_dropout_keep_prob,
                    model.dropout_keep_prob: FLAGS.dropout_keep_prob
                }
                _, step, summaries, loss, accuracy = sess.run(
                    [train_op, global_step, train_summary_op, model.loss, model.accuracy], feed_dict)
                train_summary_writer.add_summary(summaries, step)
                accs.append(accuracy)
                losses.append(loss)

                # Training log display
                if step % FLAGS.display_every == 0:
                    time_str = datetime.datetime.now().isoformat()
                    print("{}: step {}, loss {:g}, acc {:g}".format(time_str, step, loss, accuracy))

                if step % FLAGS.evaluate_every == 0:
                    acc = np.mean(accs)
                    los = np.mean(losses)
                    accs = []
                    losses = []
                    time_str = datetime.datetime.now().isoformat()
                    print("\nEvaluation: {}: step {}, loss {:g}, acc {:g}".format(time_str, step, los, acc))
                    # Model checkpoint
                    if best_accuracy < acc:
                        best_accuracy = acc
                        path = saver.save(sess, checkpoint_prefix + "-{:g}".format(best_accuracy), global_step=step)
                        print("Saved model checkpoint to {}\n".format(path))
            path = saver.save(sess, checkpoint_prefix + "_final", global_step=step)
            print("Saved model checkpoint to {}\n".format(path))

    with tf.device('/cpu:0'):
        x_text, x_position, y = data_helpers.load_train_data(FLAGS.ta_data_path + 'train_ta.npy')

    print("Train set size: {}\n".format(len(y)))

    with tf.Graph().as_default():
        session_conf = tf.ConfigProto(
            allow_soft_placement=FLAGS.allow_soft_placement,
            log_device_placement=FLAGS.log_device_placement)
        session_conf.gpu_options.allow_growth = FLAGS.gpu_allow_growth
        sess = tf.Session(config=session_conf)
        with sess.as_default():
            model = AttLSTM(
                sequence_length=FLAGS.sequence_length,
                num_classes=2,
                vocab_size=FLAGS.vocab_size,
                embedding_size=FLAGS.embedding_dim,
                hidden_size=FLAGS.hidden_size,
                l2_reg_lambda=FLAGS.l2_reg_lambda)

            # Define Training procedure
            global_step = tf.Variable(0, name="global_step", trainable=False)
            optimizer = tf.train.AdadeltaOptimizer(FLAGS.learning_rate, FLAGS.decay_rate, 1e-6)
            gvs = optimizer.compute_gradients(model.loss)
            capped_gvs = [(tf.clip_by_value(grad, -1.0, 1.0), var) for grad, var in gvs]
            train_op = optimizer.apply_gradients(capped_gvs, global_step=global_step)

            # Output directory for models and summaries
            timestamp = str(int(time.time()))
            out_dir = os.path.abspath(os.path.join(os.path.curdir, "runs", timestamp))
            print("Writing to {}\n".format(out_dir))

            # Summaries for loss and accuracy
            loss_summary = tf.summary.scalar("loss", model.loss)
            acc_summary = tf.summary.scalar("accuracy", model.accuracy)

            # Train Summaries
            train_summary_op = tf.summary.merge([loss_summary, acc_summary])
            train_summary_dir = os.path.join(out_dir, "summaries", "train")
            train_summary_writer = tf.summary.FileWriter(train_summary_dir, sess.graph)

            # Validation summaries
            dev_summary_op = tf.summary.merge([loss_summary, acc_summary])
            dev_summary_dir = os.path.join(out_dir, "summaries", "val")
            dev_summary_writer = tf.summary.FileWriter(dev_summary_dir, sess.graph)

            # Checkpoint directory. Tensorflow assumes this directory already exists so we need to create it
            checkpoint_dir = os.path.abspath(os.path.join(out_dir, "checkpoints"))
            checkpoint_prefix = os.path.join(checkpoint_dir, "model_ta")
            if not os.path.exists(checkpoint_dir):
                os.makedirs(checkpoint_dir)
            saver = tf.train.Saver(tf.global_variables(), max_to_keep=FLAGS.num_checkpoints)

            # Initialize all variables
            sess.run(tf.global_variables_initializer())

            # Pre-trained word2vec
            # pretrain_W = np.load("embedding.npy")
            # sess.run(model.W_text.assign(pretrain_W))
            # print("Success to load pre-trained word2vec model!\n")

            # Generate batches
            batches = data_helpers.batch_iter(x_text, x_position, y, FLAGS.batch_size, FLAGS.final_epochs_ta)
            # Training loop. For each batch...
            best_accuracy = 0.0  # For save checkpoint(model)
            accs = []
            losses = []
            for batch in batches:
                x_text_batch, x_position_batch, y_batch = batch
                # Train
                feed_dict = {
                    model.input_text: x_text_batch,
                    model.input_position: x_position_batch,
                    model.input_y: y_batch,
                    model.mask: mask,
                    model.emb_dropout_keep_prob: FLAGS.emb_dropout_keep_prob,
                    model.rnn_dropout_keep_prob: FLAGS.rnn_dropout_keep_prob,
                    model.dropout_keep_prob: FLAGS.dropout_keep_prob
                }
                _, step, summaries, loss, accuracy = sess.run(
                    [train_op, global_step, train_summary_op, model.loss, model.accuracy], feed_dict)
                train_summary_writer.add_summary(summaries, step)
                accs.append(accuracy)
                losses.append(loss)

                # Training log display
                if step % FLAGS.display_every == 0:
                    time_str = datetime.datetime.now().isoformat()
                    print("{}: step {}, loss {:g}, acc {:g}".format(time_str, step, loss, accuracy))

                if step % FLAGS.evaluate_every == 0:
                    acc = np.mean(accs)
                    los = np.mean(losses)
                    accs = []
                    losses = []
                    time_str = datetime.datetime.now().isoformat()
                    print("\nEvaluation: {}: step {}, loss {:g}, acc {:g}".format(time_str, step, los, acc))
                    # Model checkpoint
                    if best_accuracy < acc:
                        best_accuracy = acc
                        path = saver.save(sess, checkpoint_prefix + "-{:g}".format(best_accuracy), global_step=step)
                        print("Saved model checkpoint to {}\n".format(path))
            path = saver.save(sess, checkpoint_prefix + "_final", global_step=step)
            print("Saved model checkpoint to {}\n".format(path))


def main():
    train()


if __name__ == "__main__":
    # tf.app.run()
    main()
