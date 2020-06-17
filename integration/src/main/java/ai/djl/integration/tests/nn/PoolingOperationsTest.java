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
package ai.djl.integration.tests.nn;

import ai.djl.Model;
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.index.NDIndex;
import ai.djl.ndarray.types.Shape;
import ai.djl.nn.pooling.Pool;
import ai.djl.training.DefaultTrainingConfig;
import ai.djl.training.Trainer;
import ai.djl.training.TrainingConfig;
import ai.djl.training.initializer.Initializer;
import ai.djl.training.loss.Loss;
import org.testng.Assert;
import org.testng.annotations.Test;

public class PoolingOperationsTest {

    @Test
    public void testMaxPool1D() {
        try (Model model = Model.newInstance("model")) {
            model.setBlock(Pool.maxPool1DBlock(new Shape(2)));
            // Look for a max pool value 5
            TrainingConfig config =
                    new DefaultTrainingConfig(Loss.l2Loss(model.getNDManager()))
                            .optInitializer(Initializer.ONES);
            try (Trainer trainer = model.newTrainer(config)) {
                trainer.initialize(new Shape(2, 2, 2));

                NDManager manager = trainer.getManager();
                NDArray data = manager.ones(new Shape(2, 2, 2));
                data.set(new NDIndex(1, 1, 1), 5);
                NDArray expected = manager.ones(new Shape(2, 2, 1));
                expected.set(new NDIndex(1, 1, 0), 5);
                NDArray result = trainer.forward(new NDList(data)).singletonOrThrow();
                Assert.assertEquals(result, expected);
            }
        }
    }

    @Test
    public void testMaxPool2D() {
        try (Model model = Model.newInstance("model")) {
            model.setBlock(Pool.maxPool2DBlock(new Shape(2, 2)));
            // Look for a max pool value 5
            TrainingConfig config =
                    new DefaultTrainingConfig(Loss.l2Loss(model.getNDManager()))
                            .optInitializer(Initializer.ONES);
            try (Trainer trainer = model.newTrainer(config)) {
                trainer.initialize(new Shape(2, 2, 2, 2));

                NDManager manager = trainer.getManager();
                NDArray data = manager.ones(new Shape(2, 2, 2, 2));
                data.set(new NDIndex(1, 1, 1, 1), 5);
                NDArray expected = manager.ones(new Shape(2, 2, 1, 1));
                expected.set(new NDIndex(1, 1, 0, 0), 5);
                NDArray result = trainer.forward(new NDList(data)).singletonOrThrow();
                Assert.assertEquals(result, expected);
            }
        }
    }

    @Test
    public void testMaxPool3D() {
        try (Model model = Model.newInstance("model")) {
            model.setBlock(Pool.maxPool3DBlock(new Shape(2, 2, 2)));
            // Look for a max pool value 5
            TrainingConfig config =
                    new DefaultTrainingConfig(Loss.l2Loss(model.getNDManager()))
                            .optInitializer(Initializer.ONES);
            try (Trainer trainer = model.newTrainer(config)) {
                trainer.initialize(new Shape(2, 2, 2, 2, 2));

                NDManager manager = trainer.getManager();
                NDArray data = manager.ones(new Shape(2, 2, 2, 2, 2));
                data.set(new NDIndex(1, 1, 1, 1, 1), 5);
                NDArray expected = manager.ones(new Shape(2, 2, 1, 1, 1));
                expected.set(new NDIndex(1, 1, 0, 0, 0), 5);
                NDArray result = trainer.forward(new NDList(data)).singletonOrThrow();
                Assert.assertEquals(result, expected);
            }
        }
    }

    @Test
    public void testGlobalMaxPool() {
        try (Model model = Model.newInstance("model")) {
            model.setBlock(Pool.globalMaxPool1DBlock());
            // Look for a max pool value 5
            TrainingConfig config =
                    new DefaultTrainingConfig(Loss.l2Loss(model.getNDManager()))
                            .optInitializer(Initializer.ONES);
            try (Trainer trainer = model.newTrainer(config)) {
                trainer.initialize(new Shape(2, 2, 2));

                NDManager manager = trainer.getManager();
                NDArray data = manager.ones(new Shape(2, 2, 2));
                data.set(new NDIndex(1, 1, 1), 5);
                NDArray expected = manager.ones(new Shape(2, 2, 1));
                expected.set(new NDIndex(1, 1, 0), 5);
                NDArray result = trainer.forward(new NDList(data)).singletonOrThrow();
                Assert.assertEquals(result, expected);
            }
        }
    }

    @Test
    public void testAvgPool1D() {
        try (Model model = Model.newInstance("model")) {
            model.setBlock(Pool.avgPool1DBlock(new Shape(2)));
            // Look for a average pool value 1.5
            TrainingConfig config =
                    new DefaultTrainingConfig(Loss.l2Loss(model.getNDManager()))
                            .optInitializer(Initializer.ONES);
            try (Trainer trainer = model.newTrainer(config)) {
                trainer.initialize(new Shape(2, 2, 2));

                NDManager manager = trainer.getManager();
                NDArray data = manager.ones(new Shape(2, 2, 2));
                data.set(new NDIndex(1, 1, 1), 2);
                NDArray expected = manager.ones(new Shape(2, 2, 1));
                expected.set(new NDIndex(1, 1, 0), 1.5);
                NDArray result = trainer.forward(new NDList(data)).singletonOrThrow();
                Assert.assertEquals(result, expected);
            }
        }
    }

    @Test
    public void testAvgPool2D() {
        try (Model model = Model.newInstance("model")) {
            model.setBlock(Pool.avgPool2DBlock(new Shape(2, 2)));
            // Look for a average pool value 1.25
            TrainingConfig config =
                    new DefaultTrainingConfig(Loss.l2Loss(model.getNDManager()))
                            .optInitializer(Initializer.ONES);
            try (Trainer trainer = model.newTrainer(config)) {
                trainer.initialize(new Shape(2, 2, 2, 2));

                NDManager manager = trainer.getManager();
                NDArray data = manager.ones(new Shape(2, 2, 2, 2));
                data.set(new NDIndex(1, 1, 1, 1), 2);
                NDArray expected = manager.ones(new Shape(2, 2, 1, 1));
                expected.set(new NDIndex(1, 1, 0, 0), 1.25);
                NDArray result = trainer.forward(new NDList(data)).singletonOrThrow();
                Assert.assertEquals(result, expected);
            }
        }
    }

    @Test
    public void testAvgPool3D() {
        try (Model model = Model.newInstance("model")) {
            model.setBlock(Pool.avgPool3DBlock(new Shape(2, 2, 2)));
            // Look for a average pool value 1.125
            TrainingConfig config =
                    new DefaultTrainingConfig(Loss.l2Loss(model.getNDManager()))
                            .optInitializer(Initializer.ONES);
            try (Trainer trainer = model.newTrainer(config)) {
                trainer.initialize(new Shape(2, 2, 2, 2, 2));

                NDManager manager = trainer.getManager();
                NDArray data = manager.ones(new Shape(2, 2, 2, 2, 2));
                data.set(new NDIndex(1, 1, 1, 1, 1), 2);
                NDArray expected = manager.ones(new Shape(2, 2, 1, 1, 1));
                expected.set(new NDIndex(1, 1, 0, 0, 0), 1.125);
                NDArray result = trainer.forward(new NDList(data)).singletonOrThrow();
                Assert.assertEquals(result, expected);
            }
        }
    }

    @Test
    public void testGlobalAvgPool() {
        try (Model model = Model.newInstance("model")) {
            model.setBlock(Pool.globalAvgPool1DBlock());
            // Look for a average pool value 1.5
            TrainingConfig config =
                    new DefaultTrainingConfig(Loss.l2Loss(model.getNDManager()))
                            .optInitializer(Initializer.ONES);
            try (Trainer trainer = model.newTrainer(config)) {
                trainer.initialize(new Shape(2, 2, 2));

                NDManager manager = trainer.getManager();
                NDArray data = manager.ones(new Shape(2, 2, 2));
                data.set(new NDIndex(1, 1, 1), 2);
                NDArray expected = manager.ones(new Shape(2, 2, 1));
                expected.set(new NDIndex(1, 1, 0), 1.5);
                NDArray result = trainer.forward(new NDList(data)).singletonOrThrow();
                Assert.assertEquals(result, expected);
            }
        }
    }

    @Test
    public void testLpPool1D() {
        try (Model model = Model.newInstance("model")) {
            model.setBlock(Pool.lpPool1DBlock(new Shape(2), 1));
            TrainingConfig config =
                    new DefaultTrainingConfig(Loss.l2Loss(model.getNDManager()))
                            .optInitializer(Initializer.ONES);
            try (Trainer trainer = model.newTrainer(config)) {
                trainer.initialize(new Shape(2, 2, 2));

                NDManager manager = trainer.getManager();
                NDArray data = manager.ones(new Shape(2, 2, 2));
                data.set(new NDIndex(1, 1, 1), 2);
                NDArray expected = manager.ones(new Shape(2, 2, 1));
                expected.muli(2);
                expected.set(new NDIndex(1, 1, 0), 3);
                NDArray result = trainer.forward(new NDList(data)).singletonOrThrow();
                Assert.assertEquals(result, expected);
            }
        }
    }

    @Test
    public void testLpPool2D() {
        try (Model model = Model.newInstance("model")) {
            model.setBlock(Pool.lpPool2DBlock(new Shape(2, 2), 1));
            TrainingConfig config =
                    new DefaultTrainingConfig(Loss.l2Loss(model.getNDManager()))
                            .optInitializer(Initializer.ONES);
            try (Trainer trainer = model.newTrainer(config)) {
                trainer.initialize(new Shape(2, 2, 2, 2));

                NDManager manager = trainer.getManager();
                NDArray data = manager.ones(new Shape(2, 2, 2, 2));
                data.set(new NDIndex(1, 1, 1, 1), 2);
                NDArray expected = manager.ones(new Shape(2, 2, 1, 1));
                expected.muli(4);
                expected.set(new NDIndex(1, 1, 0, 0), 5);
                NDArray result = trainer.forward(new NDList(data)).singletonOrThrow();
                Assert.assertEquals(result, expected);
            }
        }
    }

    @Test
    public void testLpPool3D() {
        try (Model model = Model.newInstance("model")) {
            model.setBlock(Pool.lpPool3DBlock(new Shape(2, 2, 2), 1));
            TrainingConfig config =
                    new DefaultTrainingConfig(Loss.l2Loss(model.getNDManager()))
                            .optInitializer(Initializer.ONES);
            try (Trainer trainer = model.newTrainer(config)) {
                trainer.initialize(new Shape(2, 2, 2, 2, 2));

                NDManager manager = trainer.getManager();
                NDArray data = manager.ones(new Shape(2, 2, 2, 2, 2));
                data.set(new NDIndex(1, 1, 1, 1, 1), 2);
                NDArray expected = manager.ones(new Shape(2, 2, 1, 1, 1));
                expected.muli(8);
                expected.set(new NDIndex(1, 1, 0, 0, 0), 9);
                NDArray result = trainer.forward(new NDList(data)).singletonOrThrow();
                Assert.assertEquals(result, expected);
            }
        }
    }

    @Test
    public void testGlobalLpPool() {
        try (Model model = Model.newInstance("model")) {
            model.setBlock(Pool.globalLpPool1DBlock(1));
            TrainingConfig config =
                    new DefaultTrainingConfig(Loss.l2Loss(model.getNDManager()))
                            .optInitializer(Initializer.ONES);
            try (Trainer trainer = model.newTrainer(config)) {
                trainer.initialize(new Shape(2, 2, 2));

                NDManager manager = trainer.getManager();
                NDArray data = manager.ones(new Shape(2, 2, 2));
                data.set(new NDIndex(1, 1, 1), 2);
                NDArray expected = manager.ones(new Shape(2, 2, 1));
                expected.muli(2);
                expected.set(new NDIndex(1, 1, 0), 3);
                NDArray result = trainer.forward(new NDList(data)).singletonOrThrow();
                Assert.assertEquals(result, expected);
            }
        }
    }
}
