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
package ai.djl.training.listener;

import ai.djl.Device;
import ai.djl.metric.Metrics;
import ai.djl.ndarray.NDList;
import ai.djl.training.Trainer;
import ai.djl.training.evaluator.Evaluator;
import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;

/**
 * {@link TrainingListener} that records evaluator results.
 *
 * <p>Results are recorded for the following stages:
 *
 * <ul>
 *   <li>{@link #TRAIN_EPOCH} - This accumulates for the whole epoch and is recorded to a metric at
 *       the end of the epoch
 *   <li>{@link #VALIDATE_EPOCH} - This accumulates for the whole validation epoch and is recorded
 *       to a metric at the end of the epoch
 * </ul>
 *
 * <p>The training and validation evaluators are saved as metrics with names that can be found using
 * {@link EvaluatorTrainingListener#metricName(Evaluator, String)}. The validation evaluators are
 * also saved as model properties with the evaluator name.
 */
public class EvaluatorTrainingListener implements TrainingListener {

    public static final String TRAIN_EPOCH = "train/epoch";
    public static final String VALIDATE_EPOCH = "validate/epoch";

    private Map<String, Float> latestEvaluations;

    /**
     * Constructs an {@link EvaluatorTrainingListener} that updates the training progress the
     * default frequency.
     *
     * <p>Current default frequency is every 5 batches.
     */
    public EvaluatorTrainingListener() {
        latestEvaluations = new ConcurrentHashMap<>();
    }

    /** {@inheritDoc} */
    @Override
    public void onEpoch(Trainer trainer) {
        Metrics metrics = trainer.getMetrics();
        for (Evaluator evaluator : trainer.getEvaluators()) {
            float trainValue = evaluator.getAccumulator(TRAIN_EPOCH);
            if (metrics != null) {
                String key = metricName(evaluator, TRAIN_EPOCH);
                metrics.addMetric(key, trainValue);
            }

            latestEvaluations.put("train_" + evaluator.getName(), trainValue);
            float validateValue = evaluator.getAccumulator(VALIDATE_EPOCH);
            latestEvaluations.put("validate_" + evaluator.getName(), validateValue);

            if (evaluator == trainer.getLoss()) {
                latestEvaluations.put("train_loss", trainValue);
                latestEvaluations.put("validate_loss", validateValue);
            }
        }
        for (Evaluator evaluator : trainer.getEvaluators()) {
            evaluator.resetAccumulator(TRAIN_EPOCH);
            evaluator.resetAccumulator(VALIDATE_EPOCH);
        }
    }

    /** {@inheritDoc} */
    @Override
    public void onTrainingBatch(Trainer trainer, BatchData batchData) {
        //long start = System.nanoTime();
        updateEvaluators(trainer, batchData, new String[] {TRAIN_EPOCH});
        //System.out.println("Update time=" + (System.nanoTime() - start));
    }

    /** {@inheritDoc} */
    @Override
    public void onValidationBatch(Trainer trainer, BatchData batchData) {
        updateEvaluators(trainer, batchData, new String[] {VALIDATE_EPOCH});
        Metrics metrics = trainer.getMetrics();
        if (metrics != null) {
            for (Evaluator evaluator : trainer.getEvaluators()) {
                String key = metricName(evaluator, VALIDATE_EPOCH);
                float value = evaluator.getAccumulator(VALIDATE_EPOCH);
                metrics.addMetric(key, value);
            }
        }
    }

    private void updateEvaluators(Trainer trainer, BatchData batchData, String[] accumulators) {
        for (Evaluator evaluator : trainer.getEvaluators()) {
            for (Device device : batchData.getLabels().keySet()) {
                NDList labels = batchData.getLabels().get(device);
                NDList predictions = batchData.getPredictions().get(device);
                for (String accumulator : accumulators) {
                    //long start = System.nanoTime();
                    evaluator.updateAccumulator(accumulator, labels, predictions);
                    //System.out.print(" Evaliuator=" + evaluator.getName() + " time=" + (System.nanoTime() - start) );
                }
            }
        }
    }

    /** {@inheritDoc} */
    @Override
    public void onTrainingBegin(Trainer trainer) {
        for (Evaluator evaluator : trainer.getEvaluators()) {
            evaluator.addAccumulator(TRAIN_EPOCH);
            evaluator.addAccumulator(VALIDATE_EPOCH);
        }
    }

    /** {@inheritDoc} */
    @Override
    public void onTrainingEnd(Trainer trainer) {}

    /**
     * Returns the metric created with the evaluator for the given stage.
     *
     * @param evaluator the evaluator to read the metric from
     * @param stage one of {@link #TRAIN_EPOCH} or {@link #VALIDATE_EPOCH}
     * @return the metric name to use
     */
    public static String metricName(Evaluator evaluator, String stage) {
        switch (stage) {
            case TRAIN_EPOCH:
                return "train_epoch_" + evaluator.getName();
            case VALIDATE_EPOCH:
                return "validate_epoch_" + evaluator.getName();
            default:
                throw new IllegalArgumentException("Invalid metric stage");
        }
    }

    /**
     * Returns the latest evaluations.
     *
     * <p>The latest evaluations are updated on each epoch.
     *
     * @return the latest evaluations
     */
    public Map<String, Float> getLatestEvaluations() {
        return latestEvaluations;
    }
}
