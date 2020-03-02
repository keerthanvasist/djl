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
package ai.djl.examples.training;

import ai.djl.Model;
import ai.djl.engine.Engine;
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.types.Shape;
import ai.djl.nn.SequentialBlock;
import ai.djl.nn.recurrent.LSTM;
import ai.djl.training.DefaultTrainingConfig;
import ai.djl.training.GradientCollector;
import ai.djl.training.Trainer;
import ai.djl.training.evaluator.Accuracy;
import ai.djl.training.initializer.XavierInitializer;
import ai.djl.training.loss.Loss;
import ai.djl.training.loss.SoftmaxCrossEntropyLoss;
import ai.djl.util.Utils;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

public final class LSTMOperatorTest {
    private LSTMOperatorTest() {}

    public static void main(String[] args) {
        long batch = 1;
        long time = 6;
        long channel = 6;
        long state = 8;
        SequentialBlock block = new SequentialBlock();
        block.add(
                new LSTM.Builder()
                        .setStateSize((int) state)
                        .setNumStackedLayers(2)
                        .optDropRate(0)
                        .build());
        Logger logger = LoggerFactory.getLogger(LSTMOperatorTest.class);
        try (Model model = Model.newInstance()) {
            System.out.println("Engine version: " + Engine.getInstance().getVersion());
            model.setBlock(block);
            NDManager manager = model.getNDManager();
            Engine.getInstance().setRandomSeed(1234);
            NDList data = new NDList(manager.randomUniform(0, 10, new Shape(batch, time, channel)));
            Engine.getInstance().setRandomSeed(1234);
            NDList labels = new NDList(manager.randomUniform(0, 1, new Shape(batch, time, state)));

            try (Trainer trainer = model.newTrainer(setupTrainingConfig(batch))) {
                try (GradientCollector collector = trainer.newGradientCollector()) {
                    trainer.initialize(new Shape(batch, time, channel));
                    NDList preds = trainer.forward(data);
                    logger.info("\n=====OUTPUT=====\n");
                    Utils.checkNDArrayValues(preds.head(), logger, "OUTPUT");
                    Loss loss = new SoftmaxCrossEntropyLoss("loss", 1, -1, false, true);
                    NDArray lossValue = loss.evaluate(labels, preds);
                    logger.info("Loss={}", lossValue.toFloatArray()[0]);
                    collector.backward(lossValue);
                    logger.info("\n=====BACKWARD=====\n");
                    Utils.checkParameterValues(model.getBlock().getParameters(), true, logger);
                }
            }
        }
    }

    public static DefaultTrainingConfig setupTrainingConfig(long batchSize) {
        return new DefaultTrainingConfig(Loss.softmaxCrossEntropyLoss())
                .addEvaluator(new Accuracy())
                .optInitializer(new XavierInitializer());
    }
}
