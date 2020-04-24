package ai.djl.examples.training;

import ai.djl.MalformedModelException;
import ai.djl.Model;
import ai.djl.basicdataset.TatoebaEnglishFrenchDataset;
import ai.djl.basicdataset.utils.TextData;
import ai.djl.basicmodelzoo.nlp.SimpleSequenceDecoder;
import ai.djl.basicmodelzoo.nlp.SimpleSequenceEncoder;
import ai.djl.inference.Predictor;
import ai.djl.modality.nlp.EncoderDecoder;
import ai.djl.modality.nlp.embedding.TrainableTextEmbedding;
import ai.djl.modality.nlp.embedding.TrainableWordEmbedding;
import ai.djl.modality.nlp.preprocess.LowerCaseConvertor;
import ai.djl.modality.nlp.preprocess.PunctuationSeparator;
import ai.djl.modality.nlp.preprocess.SimpleTokenizer;
import ai.djl.modality.nlp.preprocess.TextTerminator;
import ai.djl.modality.nlp.preprocess.TextTruncator;
import ai.djl.modality.nlp.translator.Seq2SeqTranslator;
import ai.djl.ndarray.types.Shape;
import ai.djl.nn.Block;
import ai.djl.nn.recurrent.LSTM;
import ai.djl.training.dataset.Dataset;
import ai.djl.training.util.ProgressBar;
import ai.djl.translate.PaddingStackBatchifier;
import ai.djl.translate.TranslateException;

import java.io.IOException;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Locale;

public class TempClass {
    public static void main(String[] args)
            throws IOException, MalformedModelException, TranslateException {
        try (Model model = Model.newInstance()) {
            Path modelPath = Paths.get("/Users/kvasist");
            System.out.println(modelPath.toAbsolutePath());
            model.setBlock(getSeq2SeqModel(6470));
            model.load(modelPath, "seq2seq-finaltest");

            TatoebaEnglishFrenchDataset tatoebaEnglishFrenchDataset =
                    TatoebaEnglishFrenchDataset.builder()
                            .setSampling(10, true, false)
                            .setSourceConfiguration(
                                    new TextData.Configuration().setEmbeddingSize(32).setTrainEmbedding(true))
                            .setTargetConfiguration(
                                    new TextData.Configuration().setEmbeddingSize(32).setTrainEmbedding(true).setTextProcessors(Arrays.asList(new SimpleTokenizer(),
                                            new LowerCaseConvertor(Locale.FRENCH),
                                            new PunctuationSeparator(),
                                            new TextTruncator(8),
                                            new TextTerminator())))
                            .optUsage(Dataset.Usage.TEST)
                            .optLimit(40)
                            .build();
            tatoebaEnglishFrenchDataset.prepare(new ProgressBar());
            List<String> sentences = new ArrayList<>();
            for (int i = 0; i < 1; i++) {
                sentences.add(tatoebaEnglishFrenchDataset.getRawText(4, true));
            }
            try (Predictor<String, String> predictor = model.newPredictor(new Seq2SeqTranslator())) {
                System.out.println("English: " + sentences);
                System.out.println("French: ");
                predictor.batchPredict(sentences).forEach(System.out::println);
            }
        }
    }

    private static Block getSeq2SeqModel(int vocabSize) {
        TrainableTextEmbedding sourceEmbedding = new TrainableTextEmbedding(TrainableWordEmbedding.builder().build());
        TrainableTextEmbedding targetEmbedding = new TrainableTextEmbedding(TrainableWordEmbedding.builder().build());
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
}
