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
package ai.djl.training.loss;

import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.NDManager;
import ai.djl.training.evaluator.Evaluator;
import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;

/**
 * Loss functions (or Cost functions) are used to evaluate the model predictions against true labels
 * for optimization.
 *
 * <p>Although all evaluators can be used to measure the performance of a model, not all of them are
 * suited to being used by an optimizer. Loss functions are usually non-negative where a larger loss
 * represents worse performance. They are also real-valued to accurately compare models.
 *
 * <p>When creating a loss function, you should avoid having the loss depend on the batch size. For
 * example, if you have a loss per item in a batch and sum those losses, your loss would be {@code
 * numItemsInBatch*avgLoss}. Instead, you should take the mean of those losses to reduce out the
 * batchSize factor. Otherwise, it can make it difficult to tune the learning rate since any change
 * in the batch size would throw it off. If you have a variable batch size, it would be even more
 * difficult.
 *
 * <p>For more details about the class internals, see {@link Evaluator}.
 */
public abstract class Loss extends Evaluator {

    private Map<String, NDArray> totalLoss;
    private NDArray nanCheck;
    private NDManager manager;

    /**
     * Base class for metric with abstract update methods.
     *
     * @param manager an {@link NDManager}
     * @param name The display name of the Loss
     */
    public Loss(NDManager manager, String name) {
        super(name);
        this.manager = manager;
        totalLoss = new ConcurrentHashMap<>();
        nanCheck = manager.create(0f);
    }

    /**
     * Returns a new instance of {@link L1Loss} with default weight and batch axis.
     *
     * @param manager an {@link NDManager}
     * @return a new instance of {@link L1Loss}
     */
    public static L1Loss l1Loss(NDManager manager) {
        return new L1Loss(manager);
    }

    /**
     * Returns a new instance of {@link L1Loss} with default weight and batch axis.
     *
     * @param manager an {@link NDManager}
     * @param name the name of the loss
     * @return a new instance of {@link L1Loss}
     */
    public static L1Loss l1Loss(NDManager manager, String name) {
        return new L1Loss(manager, name);
    }

    /**
     * Returns a new instance of {@link L1Loss} with given weight and batch axis.
     *
     * @param manager an {@link NDManager}
     * @param name the name of the loss
     * @param weight the weight to apply on loss value, default 1
     * @return a new instance of {@link L1Loss}
     */
    public static L1Loss l1Loss(NDManager manager, String name, float weight) {
        return new L1Loss(manager, name, weight);
    }

    /**
     * Returns a new instance of {@link L2Loss} with default weight and batch axis.
     *
     * @param manager an {@link NDManager}
     * @return a new instance of {@link L2Loss}
     */
    public static L2Loss l2Loss(NDManager manager) {
        return new L2Loss(manager);
    }

    /**
     * Returns a new instance of {@link L2Loss} with default weight and batch axis.
     *
     * @param manager an {@link NDManager}
     * @param name the name of the loss
     * @return a new instance of {@link L2Loss}
     */
    public static L2Loss l2Loss(NDManager manager, String name) {
        return new L2Loss(manager, name);
    }

    /**
     * Returns a new instance of {@link L2Loss} with given weight and batch axis.
     *
     * @param manager an {@link NDManager}
     * @param name the name of the loss
     * @param weight the weight to apply on loss value, default 1
     * @return a new instance of {@link L2Loss}
     */
    public static L2Loss l2Loss(NDManager manager, String name, float weight) {
        return new L2Loss(manager, name, weight);
    }

    /**
     * Returns a new instance of {@link SigmoidBinaryCrossEntropyLoss} with default arguments.
     *
     * @param manager an {@link NDManager}
     * @return a new instance of {@link SigmoidBinaryCrossEntropyLoss}
     */
    public static SigmoidBinaryCrossEntropyLoss sigmoidBinaryCrossEntropyLoss(NDManager manager) {
        return new SigmoidBinaryCrossEntropyLoss(manager);
    }

    /**
     * Returns a new instance of {@link SigmoidBinaryCrossEntropyLoss} with default arguments.
     *
     * @param manager an {@link NDManager}
     * @param name the name of the loss
     * @return a new instance of {@link SigmoidBinaryCrossEntropyLoss}
     */
    public static SigmoidBinaryCrossEntropyLoss sigmoidBinaryCrossEntropyLoss(
            NDManager manager, String name) {
        return new SigmoidBinaryCrossEntropyLoss(manager, name);
    }

    /**
     * Returns a new instance of {@link SigmoidBinaryCrossEntropyLoss} with the given arguments.
     *
     * @param manager an {@link NDManager}
     * @param name the name of the loss
     * @param weight the weight to apply on the loss value, default 1
     * @param fromSigmoid whether the input is from the output of sigmoid, default false
     * @return a new instance of {@link SigmoidBinaryCrossEntropyLoss}
     */
    public static SigmoidBinaryCrossEntropyLoss sigmoidBinaryCrossEntropyLoss(
            NDManager manager, String name, float weight, boolean fromSigmoid) {
        return new SigmoidBinaryCrossEntropyLoss(manager, name, weight, fromSigmoid);
    }

    /**
     * Returns a new instance of {@link SoftmaxCrossEntropyLoss} with default arguments.
     *
     * @param manager an {@link NDManager}
     * @return a new instance of {@link SoftmaxCrossEntropyLoss}
     */
    public static SoftmaxCrossEntropyLoss softmaxCrossEntropyLoss(NDManager manager) {
        return new SoftmaxCrossEntropyLoss(manager);
    }

    /**
     * Returns a new instance of {@link SoftmaxCrossEntropyLoss} with default arguments.
     *
     * @param manager an {@link NDManager}
     * @param name the name of the loss
     * @return a new instance of {@link SoftmaxCrossEntropyLoss}
     */
    public static SoftmaxCrossEntropyLoss softmaxCrossEntropyLoss(NDManager manager, String name) {
        return new SoftmaxCrossEntropyLoss(manager, name);
    }

    /**
     * Returns a new instance of {@link SoftmaxCrossEntropyLoss} with the given arguments.
     *
     * @param manager an {@link NDManager}
     * @param name the name of the loss
     * @param weight the weight to apply on the loss value, default 1
     * @param classAxis the axis that represents the class probabilities, default -1
     * @param sparseLabel whether labels are integer array or probabilities, default true
     * @param fromLogit whether labels are log probabilities or un-normalized numbers
     * @return a new instance of {@link SoftmaxCrossEntropyLoss}
     */
    public static SoftmaxCrossEntropyLoss softmaxCrossEntropyLoss(
            NDManager manager,
            String name,
            float weight,
            int classAxis,
            boolean sparseLabel,
            boolean fromLogit) {
        return new SoftmaxCrossEntropyLoss(
                manager, name, weight, classAxis, sparseLabel, fromLogit);
    }

    /**
     * Returns a new instance of {@link MaskedSoftmaxCrossEntropyLoss} with default arguments.
     *
     * @param manager an {@link NDManager}
     * @return a new instance of {@link MaskedSoftmaxCrossEntropyLoss}
     */
    public static MaskedSoftmaxCrossEntropyLoss maskedSoftmaxCrossEntropyLoss(NDManager manager) {
        return new MaskedSoftmaxCrossEntropyLoss(manager);
    }

    /**
     * Returns a new instance of {@link MaskedSoftmaxCrossEntropyLoss} with default arguments.
     *
     * @param manager an {@link NDManager}
     * @param name the name of the loss
     * @return a new instance of {@link MaskedSoftmaxCrossEntropyLoss}
     */
    public static MaskedSoftmaxCrossEntropyLoss maskedSoftmaxCrossEntropyLoss(
            NDManager manager, String name) {
        return new MaskedSoftmaxCrossEntropyLoss(manager, name);
    }

    /**
     * Returns a new instance of {@link MaskedSoftmaxCrossEntropyLoss} with the given arguments.
     *
     * @param manager an {@link NDManager}
     * @param name the name of the loss
     * @param weight the weight to apply on the loss value, default 1
     * @param classAxis the axis that represents the class probabilities, default -1
     * @param sparseLabel whether labels are integer array or probabilities, default true
     * @param fromLogit whether labels are log probabilities or un-normalized numbers
     * @return a new instance of {@link MaskedSoftmaxCrossEntropyLoss}
     */
    public static MaskedSoftmaxCrossEntropyLoss maskedSoftmaxCrossEntropyLoss(
            NDManager manager,
            String name,
            float weight,
            int classAxis,
            boolean sparseLabel,
            boolean fromLogit) {
        return new MaskedSoftmaxCrossEntropyLoss(
                manager, name, weight, classAxis, sparseLabel, fromLogit);
    }

    /**
     * Returns a new instance of {@link HingeLoss} with default arguments.
     *
     * @param manager an {@link NDManager}
     * @return a new instance of {@link HingeLoss}
     */
    public static HingeLoss hingeLoss(NDManager manager) {
        return new HingeLoss(manager);
    }

    /**
     * Returns a new instance of {@link HingeLoss} with default arguments.
     *
     * @param manager an {@link NDManager}
     * @param name the name of the loss
     * @return a new instance of {@link HingeLoss}
     */
    public static HingeLoss hingeLoss(NDManager manager, String name) {
        return new HingeLoss(manager, name);
    }

    /**
     * Returns a new instance of {@link HingeLoss} with the given arguments.
     *
     * @param manager an {@link NDManager}
     * @param name the name of the loss
     * @param margin the margin in hinge loss. Defaults to 1.0
     * @param weight the weight to apply on loss value, default 1
     * @return a new instance of {@link HingeLoss}
     */
    public static HingeLoss hingeLoss(NDManager manager, String name, int margin, float weight) {
        return new HingeLoss(manager, name, margin, weight);
    }

    /** {@inheritDoc} */
    @Override
    public void addAccumulator(String key) {
        totalInstances.put(key, 0L);
        totalLoss.put(key, manager.create(0f));
    }

    /** {@inheritDoc} */
    @Override
    public void updateAccumulator(String key, NDList labels, NDList predictions) {
        // this is a synchronized operation, only call it at end of batch or epoch
        NDArray update = evaluate(labels, predictions).sum();
        totalInstances.computeIfPresent(key, (k, v) -> v + 1);
        totalLoss.computeIfPresent(key, (k, v) -> v.addi(update));
        nanCheck.addi(update);
    }

    /** {@inheritDoc} */
    @Override
    public void resetAccumulator(String key) {
        totalInstances.computeIfPresent(key, (k, v) -> 0L);
        totalLoss.computeIfPresent(key, (k, v) -> v.muli(0));
    }

    /** {@inheritDoc} */
    @Override
    public float getAccumulator(String key) {
        Long total = totalInstances.get(key);
        NDArray loss = totalLoss.get(key);
        if (total == null) {
            throw new IllegalArgumentException("No evaluator found at that path");
        }

        if (total == 0) {
            return Float.NaN;
        }

        return loss.getFloat() / total;
    }

    /**
     * Returns the cumulated loss value from the last time this method was called.
     *
     * @return the accumulated loss value
     * @throws IllegalArgumentException if no accumulator was added with the given key
     */
    public boolean isNan() {
        boolean isNan = Float.isNaN(nanCheck.getFloat());
        nanCheck.muli(0);
        return isNan;
    }
}
