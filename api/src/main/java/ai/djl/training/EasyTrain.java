/*
 * Copyright 2019 Amazon.com, Inc. or its affiliates. All Rights Reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License"). You may not use this file except in compliance
 * with the License. A copy of the License is located at
 *
 * http://aws.amazon.com/apache2.0/
 *
 * or in the "license" file accompanying this file. This file is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES
 * OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions
 * and limitations under the License.
 */
package ai.djl.training;

import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDList;
import ai.djl.training.dataset.Batch;
import ai.djl.training.dataset.Dataset;
import ai.djl.training.listener.TrainingListener.BatchData;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.concurrent.ConcurrentHashMap;

/** Helper for easy training of a whole model, a trainining batch, or a validation batch. */
public final class EasyTrain {
    private static Logger logger = LoggerFactory.getLogger(EasyTrain.class);
    private static long averagebatchListener = 0;
    private static long j = 0;
    private EasyTrain() {}

    /**
     * Runs a basic epoch training experience with a given trainer.
     *
     * @param trainer the trainer to train for
     * @param numEpoch the number of epochs to train
     * @param trainingDataset the dataset to train on
     * @param validateDataset the dataset to validate against. Can be null for no validation
     */
    public static void fit(
            Trainer trainer, int numEpoch, Dataset trainingDataset, Dataset validateDataset) {
        long averageStep = 0;
        long averageTrain = 0;
        long averageBatchLoad = 0;
        long averageBatchClose = 0;
        long averageListener = 0;
        long start = System.nanoTime();
        long i = 0;
        for (int epoch = 0; epoch < numEpoch; epoch++) {
            for (Batch batch : trainer.iterateDataset(trainingDataset)) {
                averageBatchLoad = (averageBatchLoad * i + (System.nanoTime() - start)) / (i + 1);
                start = System.nanoTime();
                trainBatch(trainer, batch);
                averageTrain = (averageTrain * i + (System.nanoTime() - start)) / (i + 1);
                start = System.nanoTime();
                trainer.step();
                averageStep = (averageStep * i + (System.nanoTime() - start)) / (i + 1);
                start = System.nanoTime();
                batch.close();
                averageBatchClose = (averageBatchClose * i + (System.nanoTime() - start)) / (i + 1);
                start = System.nanoTime();
                i++;
            }

            validateDataset = null;
            if (validateDataset != null) {
                for (Batch batch : trainer.iterateDataset(validateDataset)) {
                    validateBatch(trainer, batch);
                    batch.close();
                }
            }
            // reset training and validation evaluators at end of epoch
            trainer.notifyListeners(listener -> listener.onEpoch(trainer));
            averageListener = (averageListener * epoch + (System.nanoTime() - start)) / (epoch + 1);
            logger.info("Averages: load={}, train={}, batch_listener={}, step={}, close={} listener={}",
                    averageBatchLoad, averageTrain, averagebatchListener, averageStep, averageBatchClose,
                    averageListener);
            start = System.nanoTime();

        }
    }

    /**
     * Trains the model with one iteration of the given {@link Batch} of data.
     *
     * @param trainer the trainer to validate the batch with
     * @param batch a {@link Batch} that contains data, and its respective labels
     * @throws IllegalArgumentException if the batch engine does not match the trainer engine
     */
    public static void trainBatch(Trainer trainer, Batch batch) {
        if (trainer.getManager().getEngine() != batch.getManager().getEngine()) {
            throw new IllegalArgumentException(
                    "The data must be on the same engine as the trainer. You may need to change one of your NDManagers.");
        }
        Batch[] splits = batch.split(trainer.getDevices(), false);
        BatchData batchData =
                new BatchData(batch, new ConcurrentHashMap<>(), new ConcurrentHashMap<>());
        try (GradientCollector collector = trainer.newGradientCollector()) {
            for (Batch split : splits) {
                NDList data = split.getData();
                NDList labels = split.getLabels();
                NDList preds = trainer.forward(data, labels);
                long time = System.nanoTime();
                NDArray lossValue = trainer.getLoss().evaluate(labels, preds);
                collector.backward(lossValue);
                trainer.addMetric("backward", time);
                time = System.nanoTime();
                batchData.getLabels().put(labels.get(0).getDevice(), labels);
                batchData.getPredictions().put(preds.get(0).getDevice(), preds);
                trainer.addMetric("training-metrics", time);
            }
        }
        long start = System.nanoTime();
        trainer.notifyListeners(listener -> listener.onTrainingBatch(trainer, batchData));
        averagebatchListener = (averagebatchListener * j + (System.nanoTime() - start)) / (j + 1);
        j++;
    }

    /**
     * Validates the given batch of data.
     *
     * <p>During validation, the evaluators and losses are computed, but gradients aren't computed,
     * and parameters aren't updated.
     *
     * @param trainer the trainer to validate the batch with
     * @param batch a {@link Batch} of data
     * @throws IllegalArgumentException if the batch engine does not match the trainer engine
     */
    public static void validateBatch(Trainer trainer, Batch batch) {
        if (trainer.getManager().getEngine() != batch.getManager().getEngine()) {
            throw new IllegalArgumentException(
                    "The data must be on the same engine as the trainer. You may need to change one of your NDManagers.");
        }
        Batch[] splits = batch.split(trainer.getDevices(), false);
        BatchData batchData =
                new BatchData(batch, new ConcurrentHashMap<>(), new ConcurrentHashMap<>());
        for (Batch split : splits) {
            NDList data = split.getData();
            NDList labels = split.getLabels();
            NDList preds = trainer.forward(data, labels);
            batchData.getLabels().put(labels.get(0).getDevice(), labels);
            batchData.getPredictions().put(preds.get(0).getDevice(), preds);
        }

        trainer.notifyListeners(listener -> listener.onValidationBatch(trainer, batchData));
    }
}
