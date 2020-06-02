package ai.djl.examples.training;

import ai.djl.MalformedModelException;
import ai.djl.Model;
import ai.djl.basicdataset.TatoebaEnglishFrenchDataset;
import ai.djl.basicdataset.utils.TextData;
import ai.djl.basicmodelzoo.nlp.SimpleTextDecoder;
import ai.djl.basicmodelzoo.nlp.SimpleTextEncoder;
import ai.djl.inference.Predictor;
import ai.djl.modality.nlp.EncoderDecoder;
import ai.djl.modality.nlp.Vocabulary;
import ai.djl.modality.nlp.embedding.TrainableTextEmbedding;
import ai.djl.modality.nlp.embedding.TrainableWordEmbedding;
import ai.djl.modality.nlp.preprocess.LowerCaseConvertor;
import ai.djl.modality.nlp.preprocess.PunctuationSeparator;
import ai.djl.modality.nlp.preprocess.SimpleTokenizer;
import ai.djl.modality.nlp.preprocess.TextTerminator;
import ai.djl.modality.nlp.preprocess.TextTruncator;
import ai.djl.modality.nlp.translator.SimpleText2TextTranslator;
import ai.djl.nn.Block;
import ai.djl.nn.recurrent.LSTM;
import ai.djl.training.dataset.Dataset;
import ai.djl.training.util.ProgressBar;
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
        try (Model model = Model.newInstance("Seq2Seq")) {
            Path modelPath = Paths.get("examples/build/model");
            System.out.println(modelPath.toAbsolutePath());
            TatoebaEnglishFrenchDataset tatoebaEnglishFrenchDataset =
                    TatoebaEnglishFrenchDataset.builder()
                            .setSampling(10, true, false)
                            .setSourceConfiguration(
                                    new TextData.Configuration().setEmbeddingSize(32))
                            .setTargetConfiguration(
                                    new TextData.Configuration()
                                            .setEmbeddingSize(32)
                                            .setTextProcessors(
                                                    Arrays.asList(
                                                            new SimpleTokenizer(),
                                                            new LowerCaseConvertor(Locale.FRENCH),
                                                            new PunctuationSeparator(),
                                                            new TextTruncator(8),
                                                            new TextTerminator())))
                            .optUsage(Dataset.Usage.TEST)
                            .optLimit(40)
                            .build();
            tatoebaEnglishFrenchDataset.prepare(new ProgressBar());
            model.setBlock(getSeq2SeqModel(tatoebaEnglishFrenchDataset.getVocabulary(false).getAllTokens().size()));
            model.load(modelPath, "seq2seqMTEn-Fr");


            List<String> englishSentences = new ArrayList<>();
            List<String> frenchSentences = new ArrayList<>();
            try (Predictor<String, String> predictor =
                         model.newPredictor(new SimpleText2TextTranslator())) {
                for (int i = 40; i < 50; i++) {
                    englishSentences.add(tatoebaEnglishFrenchDataset.getRawText(i, true));
                    frenchSentences.add(
                            predictor.predict(tatoebaEnglishFrenchDataset.getRawText(i, true)));
                }
                System.out.print("English: ");
                for (int i = 0; i < englishSentences.size(); i++) {
                    System.out.print(i + ". " + englishSentences.get(i) + "\t");
                }
                System.out.println();
                System.out.print("French alone: ");
                for (int i = 0; i < frenchSentences.size(); i++) {
                    System.out.print(i + ". " + frenchSentences.get(i) + "\t");
                }

                frenchSentences = predictor.batchPredict(englishSentences);
                System.out.println();
                System.out.print("French batch: ");
                for (int i = 0; i < frenchSentences.size(); i++) {
                    System.out.print(i + ". " + frenchSentences.get(i) + "\t");
                }
            }
        }
    }

    private static Block getSeq2SeqModel(int vocabSize) {
        TrainableTextEmbedding sourceEmbedding =
                new TrainableTextEmbedding(TrainableWordEmbedding.builder().build());
        TrainableTextEmbedding targetEmbedding =
                new TrainableTextEmbedding(TrainableWordEmbedding.builder().build());
        SimpleTextEncoder simpleSequenceEncoder =
                new SimpleTextEncoder(
                        sourceEmbedding,
                        new LSTM.Builder()
                                .setStateSize(32)
                                .setNumStackedLayers(2)
                                .optDropRate(0)
                                .build());
        SimpleTextDecoder simpleSequenceDecoder =
                new SimpleTextDecoder(
                        targetEmbedding,
                        new LSTM.Builder()
                                .setStateSize(32)
                                .setNumStackedLayers(2)
                                .optDropRate(0)
                                .build(),
                        vocabSize);
        return new EncoderDecoder(simpleSequenceEncoder, simpleSequenceDecoder);
    }
}
