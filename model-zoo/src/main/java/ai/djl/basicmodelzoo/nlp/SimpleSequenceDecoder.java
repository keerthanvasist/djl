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
package ai.djl.basicmodelzoo.nlp;

import ai.djl.modality.nlp.Decoder;
import ai.djl.modality.nlp.embedding.TrainableTextEmbedding;
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDArrays;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.types.DataType;
import ai.djl.ndarray.types.Shape;
import ai.djl.nn.Block;
import ai.djl.nn.SequentialBlock;
import ai.djl.nn.core.Linear;
import ai.djl.nn.recurrent.RecurrentBlock;
import ai.djl.training.ParameterStore;

import java.util.ArrayList;
import java.util.List;

/**
 * {@code SimpleSequenceDecoder} implements a {@link Decoder} that employs a {@link RecurrentBlock}
 * to decode text input.
 */
public class SimpleSequenceDecoder extends Decoder {
    private RecurrentBlock recurrentBlock;
    private TrainableTextEmbedding trainableTextEmbedding;
    private String bosToken;
    private String eosToken;

    /**
     * Contructs a new instance of {@code SimpleSequenceDecoder} with the given {@link
     * RecurrentBlock}. Use this constructor if you are planning to use pre-trained embeddings that
     * don't need further training.
     *
     * @param recurrentBlock the recurrent block to be used to decode
     * @param vocabSize the size of the {@link ai.djl.modality.nlp.Vocabulary}
     */
    public SimpleSequenceDecoder(
            RecurrentBlock recurrentBlock, int vocabSize,  String bosToken, String eosToken) {
        this(null, recurrentBlock, vocabSize, bosToken, eosToken);
    }

    /**
     * Contructs a new instance of {@code SimpleSequenceDecoder} with the given {@link
     * RecurrentBlock}. Use this constructor if you are planning to use pre-trained embeddings that
     * don't need further training.
     *
     * @param trainableTextEmbedding the {@link TrainableTextEmbedding} to train embeddings with
     * @param recurrentBlock the recurrent block to be used to decode
     * @param vocabSize the size of the {@link ai.djl.modality.nlp.Vocabulary}
     */
    public SimpleSequenceDecoder(
            TrainableTextEmbedding trainableTextEmbedding,
            RecurrentBlock recurrentBlock,
            int vocabSize,
            String bosToken,
            String eosToken) {
        super(getBlock(trainableTextEmbedding, recurrentBlock, vocabSize));
        this.recurrentBlock = recurrentBlock;
        this.trainableTextEmbedding = trainableTextEmbedding;
        this.bosToken = bosToken;
        this.eosToken = eosToken;
    }

    private static Block getBlock(
            TrainableTextEmbedding trainableTextEmbedding,
            RecurrentBlock recurrentBlock,
            int vocabSize) {
        SequentialBlock sequentialBlock = new SequentialBlock();
        sequentialBlock
                .add(trainableTextEmbedding)
                .add(recurrentBlock)
                .add(Linear.builder().setOutChannels(vocabSize).optFlatten(false).build());
        return sequentialBlock;
    }

    /** {@inheritDoc} */
    @Override
    public void initState(NDList encoderState) {
        recurrentBlock.setBeginStates(encoderState);
    }

    /** {@inheritDoc} */
    @Override
    public NDList predict(ParameterStore parameterStore, NDList inputs) {
        Shape inputShape = inputs.get(0).getShape();
        if (inputShape.get(1) != 1) {
            throw new IllegalArgumentException(
                    "Input sequence length must be 1 during prediction");
        }
        NDList output = new NDList();
        for (int i = 0; i < 10; i++) {
            inputs = block.predict(parameterStore, inputs);
            inputs = new NDList(inputs.head().argMax(2));
            output.add(inputs.head().transpose(1, 0));
        }
        NDList ret = new NDList(NDArrays.stack(output).transpose(2, 1, 0));
        return new NDList(NDArrays.stack(output).transpose(2, 1, 0));
    }
}
