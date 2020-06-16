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
package ai.djl.basicdataset;

import ai.djl.basicdataset.utils.TextData.Configuration;
import ai.djl.modality.nlp.embedding.EmbeddingException;
import ai.djl.ndarray.NDManager;
import ai.djl.training.dataset.Record;
import java.io.IOException;
import org.testng.Assert;
import org.testng.annotations.Test;

public class TatoebaEnglishFrenchDatasetTest {

    private static final int EMBEDDING_SIZE = 15;

    @Test
    public void testGetDataWithPreTrainedEmbedding() throws IOException {
        try (NDManager manager = NDManager.newBaseManager()) {
            TatoebaEnglishFrenchDataset tatoebaEnglishFrenchDataset =
                    TatoebaEnglishFrenchDataset.builder()
                            .setManager(manager)
                            .setSourceConfiguration(
                                    new Configuration()
                                            .setTextEmbedding(
                                                    TestUtils.getTextEmbedding(
                                                            manager, EMBEDDING_SIZE)))
                            .setTargetConfiguration(
                                    new Configuration()
                                            .setTextEmbedding(
                                                    TestUtils.getTextEmbedding(
                                                            manager, EMBEDDING_SIZE)))
                            .setSampling(32, true)
                            .optLimit(1000)
                            .build();
            tatoebaEnglishFrenchDataset.prepare();
            Record record = tatoebaEnglishFrenchDataset.get(manager, 0);
            Assert.assertEquals(record.getData().get(0).getShape().dimension(), 1);
            Assert.assertEquals(record.getLabels().get(0).getShape().dimension(), 1);
        }
    }

    @Test
    public void testGetDataWithTrainableEmbedding() throws IOException, EmbeddingException {
        try (NDManager manager = NDManager.newBaseManager()) {
            TatoebaEnglishFrenchDataset tatoebaEnglishFrenchDataset =
                    TatoebaEnglishFrenchDataset.builder()
                            .setManager(manager)
                            .setSourceConfiguration(
                                    new Configuration()
                                            .setTextEmbedding(
                                                    TestUtils.getTextEmbedding(
                                                            manager, EMBEDDING_SIZE))
                                            .setEmbeddingSize(EMBEDDING_SIZE))
                            .setTargetConfiguration(
                                    new Configuration()
                                            .setTextEmbedding(
                                                    TestUtils.getTextEmbedding(
                                                            manager, EMBEDDING_SIZE))
                                            .setEmbeddingSize(EMBEDDING_SIZE))
                            .setSampling(32, true)
                            .optLimit(1000)
                            .build();
            tatoebaEnglishFrenchDataset.prepare();

            Record record = tatoebaEnglishFrenchDataset.get(manager, 0);
            Assert.assertEquals(record.getData().size(), 1);
            Assert.assertEquals(record.getData().get(0).getShape().dimension(), 1);
            Assert.assertEquals(record.getLabels().size(), 1);
            Assert.assertEquals(record.getLabels().get(0).getShape().dimension(), 1);
        }
    }
}
