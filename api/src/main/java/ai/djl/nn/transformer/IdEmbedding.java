/*
 * Copyright 2020 Amazon.com, Inc. or its affiliates. All Rights Reserved.
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
package ai.djl.nn.transformer;

import ai.djl.Device;
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.types.DataType;
import ai.djl.ndarray.types.Shape;
import ai.djl.nn.AbstractBlock;
import ai.djl.nn.Block;
import ai.djl.nn.Parameter;
import ai.djl.nn.ParameterType;
import ai.djl.training.ParameterStore;
import ai.djl.util.PairList;
import java.util.Arrays;

/**
 * An Embedding from integer ids to float vectors. Output shape is the input shape + one dimension
 * for the embedding. E.g. If input shape is (-1, 128), embedding size is 1024, then the output
 * shape is (-1, 128, 1024)
 */
public final class IdEmbedding extends AbstractBlock {

    private static final byte VERSION = 1;
    private static final String EMBEDDING_PARAM_NAME = "embedding";

    private final int dictionarySize;
    private final int embeddingSize;

    private final Parameter embedding;

    private IdEmbedding(Builder builder) {
        super(VERSION);
        this.dictionarySize = builder.dictionarySize;
        this.embeddingSize = builder.embeddingSize;
        this.embedding =
                addParameter(
                        new Parameter(EMBEDDING_PARAM_NAME, this, ParameterType.WEIGHT),
                        new Shape(dictionarySize, embeddingSize));
    }

    /** {@inheritDoc} */
    @Override
    public Shape[] getOutputShapes(NDManager manager, Shape[] inputShapes) {
        return new Shape[] {inputShapes[0].addAll(new Shape(embeddingSize))};
    }

    @Override
    public NDList forward(
            ParameterStore ps, NDList inputs, boolean training, PairList<String, Object> params) {
        return forward(ps, inputs, training);
    }

    @Override
    public NDList forward(ParameterStore ps, NDList inputs, boolean training) {
        return new NDList(inputs.singletonOrThrow());
    }

    /**
     * Performs a lookup into the embedding using the given input ids.
     *
     * @param parameterStore used to get the current state of the embedding table
     * @param input an ndarry of token ids
     * @return the embeddings for the given ids
     */
    public NDArray forward(ParameterStore parameterStore, NDArray input) {
        // on info to the right shapes, see: http://beta.mxnet.io/r/api/mx.symbol.gather_nd.html
        NDArray ids = input.flatten().reshape(1, input.getShape().size());
        // create the embedding Table
        NDArray embeddingTable = parameterStore.getValue(embedding, ids.getDevice());
        // We do not perform a sparse lookup, instead we just project into the table
        NDArray result = MissingOps.gatherNd(embeddingTable, ids);
        // we want the original shape of the input + the last dimension of the embedding
        Shape targetShape = input.getShape().addAll(new Shape(embeddingTable.getShape().get(1)));
        return result.reshape(targetShape);
    }

    /**
     * Turns an array of embeddings of shape (d0 ... dN, E) into an array of log probabilities of
     * shape (d0 ... dN, D) that shows the probability distribution that a given embedding
     * corresponds to an entry in the internal embedding table.
     *
     * @param parameterStore the parameters store
     * @param input the embeddings to create log probabilities for
     * @return log probabilities for each embedding
     */
    public NDArray probabilities(ParameterStore parameterStore, NDArray input) {
        // reshape input into a matrix
        NDArray asMatrix = input.reshape(-1, embeddingSize);
        // get embedding table
        NDArray embeddingTableTransposed =
                parameterStore.getValue(embedding, input.getDevice()).transpose();
        embeddingTableTransposed.attach(input.getManager());
        // Create raw logits by taking the scalar product of the tokens and the embedding table
        NDArray logitsFlat = asMatrix.dot(embeddingTableTransposed);
        // turn the logits int negative log probabilities
        NDArray logProbsFlat = logitsFlat.logSoftmax(1);
        // turn probs back into original shape
        Shape targetShape =
                input.getShape()
                        .slice(0, input.getShape().dimension() - 1)
                        .addAll(new Shape(dictionarySize));
        return logProbsFlat.reshape(targetShape);
    }

    /**
     * Quick hack for bert model to acces embedding table, replace by a proper function to calculate
     * raw logits from embeddings. TODO: replace by function to get logits
     *
     * @param ps the parameter store
     * @param device device to get internal table for
     * @return this embedding table as an array on the given device
     */
    public NDArray getValue(ParameterStore ps, Device device) {
        return ps.getValue(embedding, device);
    }

    @Override
    public void initializeChildBlocks(NDManager manager, DataType dataType, Shape... inputShapes) {
        inputNames = Arrays.asList("tokenIds");
        // nothing else to do, we have no child blocks
    }

    /** The Builder to construct an {@link IdEmbedding} type of {@link Block}. */
    public static final class Builder {

        private int dictionarySize;
        private int embeddingSize;

        /**
         * Sets the number of ids that should be embedded. Valid ids are 0 to dictionarySize - 1.
         *
         * @param dictionarySize the number of ids that should be embedded. Valid ids are 0 to
         *     dictionarySize - 1.
         * @return this builder
         */
        public Builder setDictionarySize(final int dictionarySize) {
            this.dictionarySize = dictionarySize;
            return this;
        }

        /**
         * Sets the size of the embeddings.
         *
         * @param embeddingSize the size of the embeddings.
         * @return this builder
         */
        public Builder setEmbeddingSize(final int embeddingSize) {
            this.embeddingSize = embeddingSize;
            return this;
        }

        /**
         * Builds the {@link IdEmbedding}.
         *
         * @return the constructed {@code IdEmbedding}
         * @throws IllegalArgumentException if all required parameters (items, embeddingSize) have
         *     not been set
         */
        public IdEmbedding build() {
            if (dictionarySize <= 0) {
                throw new IllegalArgumentException(
                        "You must specify the dictionary Size for the embedding.");
            }
            if (embeddingSize == 0) {
                throw new IllegalArgumentException("You must specify the embedding size");
            }
            return new IdEmbedding(this);
        }
    }
}
