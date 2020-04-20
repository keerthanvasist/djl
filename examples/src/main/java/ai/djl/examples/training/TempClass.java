package ai.djl.examples.training;

import ai.djl.MalformedModelException;
import ai.djl.Model;
import ai.djl.basicdataset.TatoebaEnglishFrenchDataset;
import ai.djl.basicdataset.TextDataset;
import ai.djl.basicmodelzoo.nlp.SimpleSequenceDecoder;
import ai.djl.basicmodelzoo.nlp.SimpleSequenceEncoder;
import ai.djl.examples.training.util.Arguments;
import ai.djl.modality.nlp.EncoderDecoder;
import ai.djl.modality.nlp.embedding.EmbeddingException;
import ai.djl.modality.nlp.embedding.TrainableTextEmbedding;
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.types.DataType;
import ai.djl.ndarray.types.Shape;
import ai.djl.nn.Block;
import ai.djl.nn.recurrent.LSTM;
import ai.djl.training.DefaultTrainingConfig;
import ai.djl.training.dataset.Dataset;
import ai.djl.training.loss.MaskedSoftmaxCrossEntropyLoss;
import ai.djl.training.util.ProgressBar;

import java.io.IOException;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;

public class TempClass {
    public static void main(String[] args) throws IOException, EmbeddingException, MalformedModelException {
        try (Model model = Model.newInstance()) {
            Path modelPath = Paths.get("build/model");
            System.out.println(modelPath.toAbsolutePath());
            TatoebaEnglishFrenchDataset dataset = getDataset(Dataset.Usage.TRAIN, 64, Long.MAX_VALUE);
            TrainableTextEmbedding sourceTextEmbedding = (TrainableTextEmbedding) dataset.getTextEmbedding(true);
            TrainableTextEmbedding targetTextEmbedding = (TrainableTextEmbedding) dataset.getTextEmbedding(false);
            model.setBlock(getSeq2SeqModel(sourceTextEmbedding, targetTextEmbedding, dataset.getVocabulary(false).size()));
            model.load(Paths.get("build/model"), "trainSeqToSeq");
            EncoderDecoder encoderDecoder = (EncoderDecoder) model.getBlock();

            NDList predictionInput = new NDList();
            NDArray sourceIndices = sourceTextEmbedding.preprocessTextToEmbed(model.getNDManager(), Arrays.asList("I won !?".split(" ")));
            NDArray targetIndex = targetTextEmbedding.preprocessTextToEmbed(model.getNDManager(), Collections.singletonList("<bos>")).reshape(new Shape(1, 1));
            predictionInput.add(sourceIndices.reshape(1, sourceIndices.getShape().size()));
            predictionInput.add(targetIndex);

            NDList prediction = model.newTrainer(new DefaultTrainingConfig(new MaskedSoftmaxCrossEntropyLoss())).predict(predictionInput);


            List<String> translation = targetTextEmbedding.unembedText(prediction.head().flatten().toType(DataType.INT32, false));
            System.out.println(translation);

        }
    }

    public static void textEmbeddingTest() throws IOException, EmbeddingException {
        TatoebaEnglishFrenchDataset dataset = getDataset(Dataset.Usage.TRAIN, 64, Long.MAX_VALUE);
        NDArray indices = dataset.getTextEmbedding(false).preprocessTextToEmbed(NDManager.newBaseManager(), Arrays.asList("<bos> va je ? <eos>".split(" ")));
        System.out.println(indices);
        System.out.println(dataset.getTextEmbedding(false).unembedText(indices));
        System.out.println(dataset.getVocabulary(false).getAllTokens().contains("<eos>"));
    }

    private static Block getSeq2SeqModel(TrainableTextEmbedding sourceEmbedding, TrainableTextEmbedding targetEmbedding, int vocabSize) {
        SimpleSequenceEncoder simpleSequenceEncoder =
                new SimpleSequenceEncoder(
                        sourceEmbedding,
                        new LSTM.Builder()
                                .setStateSize(32)
                                .setNumStackedLayers(2)
                                .optDropRate(0)
                                .build());
        SimpleSequenceDecoder simpleSequenceDecoder =
                new SimpleSequenceDecoder(
                        targetEmbedding,
                        new LSTM.Builder()
                                .setStateSize(32)
                                .setNumStackedLayers(2)
                                .optDropRate(0)
                                .build(),
                        vocabSize,
                        "<bos>",
                        "<eos>");
        return new EncoderDecoder(simpleSequenceEncoder, simpleSequenceDecoder);
    }

    public static TatoebaEnglishFrenchDataset getDataset(Dataset.Usage usage, int batchSize, long limit)
            throws IOException {
        TatoebaEnglishFrenchDataset tatoebaEnglishFrenchDataset =
                TatoebaEnglishFrenchDataset.builder()
                        .setValidLength(true)
                        .setSampling(batchSize, true, true)
                        .optEmbeddingSize(32)
                        .optUsage(usage)
                        .optLimit(limit)
                        .build();
        tatoebaEnglishFrenchDataset.prepare(new ProgressBar());
        return tatoebaEnglishFrenchDataset;
    }
}
