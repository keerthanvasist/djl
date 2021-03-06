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
package ai.djl.pytorch.zoo;

import ai.djl.pytorch.zoo.cv.classification.Resnet;
import ai.djl.pytorch.zoo.cv.objectdetection.FasterRcnnDetectionModelLoader;
import ai.djl.repository.Repository;
import ai.djl.repository.zoo.ModelLoader;
import ai.djl.repository.zoo.ModelNotFoundException;
import ai.djl.repository.zoo.ModelZoo;
import java.lang.reflect.Field;
import java.util.ArrayList;
import java.util.List;

/**
 * PtModelZoo is a repository that contains all MXNet models in {@link
 * ai.djl.pytorch.engine.PtSymbolBlock} for DJL.
 */
public class PtModelZoo implements ModelZoo {

    public static final String NAME = "PyTorch";

    private static final String DJL_REPO_URL = "https://mlrepo.djl.ai/";
    private static final Repository REPOSITORY = Repository.newInstance("PyTorch", DJL_REPO_URL);
    public static final String GROUP_ID = "ai.djl.pytorch";

    public static final Resnet RESNET = new Resnet(REPOSITORY);
    public static final FasterRcnnDetectionModelLoader FASTER_RCNN =
            new FasterRcnnDetectionModelLoader(REPOSITORY);

    /** {@inheritDoc} */
    @Override
    public List<ModelLoader<?, ?>> getModelLoaders() {
        List<ModelLoader<?, ?>> list = new ArrayList<>();
        try {
            Field[] fields = PtModelZoo.class.getDeclaredFields();
            for (Field field : fields) {
                if (field.getType().isAssignableFrom(ModelLoader.class)) {
                    list.add((ModelLoader<?, ?>) field.get(null));
                }
            }
        } catch (ReflectiveOperationException e) {
            // ignore
        }
        return list;
    }

    /** {@inheritDoc} */
    @SuppressWarnings("unchecked")
    @Override
    public <I, O> ModelLoader<I, O> getModelLoader(String name) throws ModelNotFoundException {
        try {
            Field field = PtModelZoo.class.getDeclaredField(name);
            return (ModelLoader<I, O>) field.get(null);
        } catch (ReflectiveOperationException e) {
            throw new ModelNotFoundException(
                    "Model: " + name + " is not defined in PtModelZoo.", e);
        }
    }
}
