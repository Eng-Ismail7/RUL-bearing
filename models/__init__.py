import sys
sys.path.append(".")

from models.evaluation import do_eval, create_plots_and_latex
from util.constants import BASIC_STATISTICAL_FEATURES, ENTROPY_FEATURES, FREQUENCY_FEATURES, ALL_FEATURES, ENTROPY_FREQUENCY_FEATURES

from models.DegradationModel import DegradationModel
from models.CNNSpectraFeatures import CNNSpectraFeatures
from models.ComputedFeaturesFFNN import ComputedFeaturesFFNN
from models.CombinedFeaturesFFNN import EmbeddingFeaturesFNNN
from models.DataSetType import DataSetType
from models.HealthStageClassifier import LiEtAl2019HealthStageClassifier

from rul_features.learned_features.supervised.cnn_multiscale_features import CNNEmbedding, train_and_store_cnn
from rul_features.learned_features.unsupervised.autoencoder import AutoencoderEmbedding
from rul_features.learned_features.unsupervised.principal_component_analysis import PCAEmbedding
from rul_features.learned_features.unsupervised.isomap import IsomapEmbedding

if __name__ == '__main__':
    # Computed Features Models
    statistical_features_no_classifier_ffnn = ComputedFeaturesFFNN(feature_list=BASIC_STATISTICAL_FEATURES,
                                                                   name="statistical")
    entropy_features_no_classifier_ffnn = ComputedFeaturesFFNN(feature_list=ENTROPY_FEATURES,
                                                               name="entropy")
    frequency_entropy_features_ffnn = ComputedFeaturesFFNN(feature_list=ENTROPY_FREQUENCY_FEATURES,
                                                               name="frequency_entropy")


    frequency_pca = EmbeddingFeaturesFNNN(name="FrequencyPCA",
                                                    embedding_method=PCAEmbedding(),
                                                    encoding_size=450,
                                                    data_set_type=DataSetType.raw,
                                                    use_frequency_embedding=True)
                                                    
    frequency_autoencoder = EmbeddingFeaturesFNNN(name="FrequencyAutoencoder",
                                                    embedding_method=AutoencoderEmbedding(),
                                                    encoding_size=450,
                                                    data_set_type=DataSetType.raw,
                                                    use_frequency_embedding=True)

    frequency_features_no_classifier_ffnn = ComputedFeaturesFFNN(feature_list=FREQUENCY_FEATURES,
                                                                 name="frequency")

    computed_features_autoencoder_combiner_ffnn = EmbeddingFeaturesFNNN(name="Autoencoder combined",
                                                                        embedding_method=AutoencoderEmbedding(),
                                                                        encoding_size=25,
                                                                        data_set_type=DataSetType.computed)

    computed_features_pca_combiner_ffnn = EmbeddingFeaturesFNNN(name="PCA combined",
                                                                embedding_method=PCAEmbedding(),
                                                                encoding_size=25,
                                                                data_set_type=DataSetType.computed)

    computed_features_isomap_combiner_ffnn = EmbeddingFeaturesFNNN(name="Isomap combined",
                                                                   embedding_method=IsomapEmbedding(),
                                                                   encoding_size=25,
                                                                   data_set_type=DataSetType.computed)
    computed_features_uncombined_ffnn = ComputedFeaturesFFNN(feature_list=ALL_FEATURES,
                                                             name="Uncombined")

    frequency_embedded_pca = EmbeddingFeaturesFNNN(name="FrequencyStatisticalPCA",
                                                    embedding_method=PCAEmbedding(),
                                                    encoding_size=25,
                                                    data_set_type=DataSetType.computed,
                                                    use_frequency_embedding=True)
                                                    
    frequency_embedded_autoencoder = EmbeddingFeaturesFNNN(name="FrequencyStatisticalAutoencoder",
                                                    embedding_method=AutoencoderEmbedding(),
                                                    encoding_size=25,
                                                    data_set_type=DataSetType.computed,
                                                    use_frequency_embedding=True)
                                                    
    # Learned Features Models
    pca_features_no_classifier_ffnn = EmbeddingFeaturesFNNN(name="PCA learned",
                                                            embedding_method=PCAEmbedding(),
                                                            encoding_size=450,
                                                            data_set_type=DataSetType.raw)
    isomap_features_no_classifier_ffnn = EmbeddingFeaturesFNNN(name="Isomap learned",
                                                               embedding_method=IsomapEmbedding(),
                                                               encoding_size=450,
                                                               data_set_type=DataSetType.raw)
    autoencoder_features_no_classifier_ffnn = EmbeddingFeaturesFNNN(name="Autoencoder learned",
                                                                    embedding_method=AutoencoderEmbedding(),
                                                                    encoding_size=450,
                                                                    data_set_type=DataSetType.raw)
    cnn_features_no_classifier_ffnn = CNNSpectraFeatures(name="CNN learned")


    test_cnn_embedded = EmbeddingFeaturesFNNN(name="CNNEmbedding",
                                                embedding_method=CNNEmbedding(),
                                                encoding_size=-1,
                                                data_set_type=DataSetType.spectra)

    ##### MODELS
    
    models_dict = { 
        "A": [statistical_features_no_classifier_ffnn],
        "B": [entropy_features_no_classifier_ffnn],
        "C": [pca_features_no_classifier_ffnn],
        "D": [autoencoder_features_no_classifier_ffnn],
        "E": [computed_features_pca_combiner_ffnn],
        "F": [computed_features_autoencoder_combiner_ffnn],
        "G": [frequency_features_no_classifier_ffnn],
        "H": [frequency_entropy_features_ffnn],
        "I": [frequency_pca],
        "J": [frequency_autoencoder],
        "K": [frequency_embedded_pca],
        "L": [frequency_embedded_autoencoder],
        "M": [test_cnn_embedded]
    }
    hsc = LiEtAl2019HealthStageClassifier()
    # if none of the following porps are set FFNN is used
    use_svr = False
    use_gpr = False
    use_poly_reg = True
    do_eval(model_dict=models_dict, health_stage_classifier=hsc, use_svr=use_svr, use_gpr=use_gpr, use_poly_reg=use_poly_reg)

    models_dict = { 
        "A": [statistical_features_no_classifier_ffnn],
        "B": [entropy_features_no_classifier_ffnn],
        "C": [pca_features_no_classifier_ffnn],
        "D": [autoencoder_features_no_classifier_ffnn],
        "E": [computed_features_pca_combiner_ffnn],
        "F": [computed_features_autoencoder_combiner_ffnn],
        "G": [frequency_features_no_classifier_ffnn],
        "H": [frequency_entropy_features_ffnn],
        "I": [frequency_pca],
        "J": [frequency_autoencoder],
        "K": [frequency_embedded_pca],
        "L": [frequency_embedded_autoencoder],
        "M": [test_cnn_embedded]
    }
    hsc = None 
    # if none of the following porps are set FFNN is used
    use_svr = False
    use_gpr = False
    use_poly_reg = True
    do_eval(model_dict=models_dict, health_stage_classifier=hsc, use_svr=use_svr, use_gpr=use_gpr, use_poly_reg=use_poly_reg)

    models_dict = { 
        "A": [statistical_features_no_classifier_ffnn],
        "B": [entropy_features_no_classifier_ffnn],
        "C": [pca_features_no_classifier_ffnn],
        "D": [autoencoder_features_no_classifier_ffnn],
        "E": [computed_features_pca_combiner_ffnn],
        "F": [computed_features_autoencoder_combiner_ffnn],
        "G": [frequency_features_no_classifier_ffnn],
        "H": [frequency_entropy_features_ffnn],
        "I": [frequency_pca],              
        "J": [frequency_autoencoder],      
        "K": [frequency_embedded_pca],     
        "L": [frequency_embedded_autoencoder], 
        "M": [test_cnn_embedded] 
    }
    hsc = LiEtAl2019HealthStageClassifier() # None 
    # if none of the following porps are set FFNN is used
    use_svr = True
    use_gpr = False
    use_poly_reg = False
    do_eval(model_dict=models_dict, health_stage_classifier=hsc, use_svr=use_svr, use_gpr=use_gpr, use_poly_reg=use_poly_reg)

    models_dict = { 
        "A": [statistical_features_no_classifier_ffnn],
        "B": [entropy_features_no_classifier_ffnn],
        "C": [pca_features_no_classifier_ffnn],
        "D": [autoencoder_features_no_classifier_ffnn],
        "E": [computed_features_pca_combiner_ffnn],
        "F": [computed_features_autoencoder_combiner_ffnn],
        "G": [frequency_features_no_classifier_ffnn],
        "H": [frequency_entropy_features_ffnn],
        "I": [frequency_pca],              
        "J": [frequency_autoencoder],      
        "K": [frequency_embedded_pca],     
        "L": [frequency_embedded_autoencoder],     
        "M": [test_cnn_embedded]
    }
    hsc = None 
    # if none of the following porps are set FFNN is used
    use_svr = True
    use_gpr = False
    use_poly_reg = False
    do_eval(model_dict=models_dict, health_stage_classifier=hsc, use_svr=use_svr, use_gpr=use_gpr, use_poly_reg=use_poly_reg)

    models_dict = { 
        "A": [statistical_features_no_classifier_ffnn],
        "B": [entropy_features_no_classifier_ffnn],
        "C": [pca_features_no_classifier_ffnn],
        "D": [autoencoder_features_no_classifier_ffnn],
        "E": [computed_features_pca_combiner_ffnn],
        "F": [computed_features_autoencoder_combiner_ffnn],
        "G": [frequency_features_no_classifier_ffnn],
        "H": [frequency_entropy_features_ffnn],
        "I": [frequency_pca],
        "J": [frequency_autoencoder],
        "K": [frequency_embedded_pca],
        "L": [frequency_embedded_autoencoder],
        "M": [test_cnn_embedded]
    }
    hsc = LiEtAl2019HealthStageClassifier() 
    # if none of the following porps are set FFNN is used
    use_svr = False
    use_gpr = False
    use_poly_reg = False
    do_eval(model_dict=models_dict, health_stage_classifier=hsc, use_svr=use_svr, use_gpr=use_gpr, use_poly_reg=use_poly_reg)

    models_dict = { 
        "A": [statistical_features_no_classifier_ffnn],
        "B": [entropy_features_no_classifier_ffnn],
        "C": [pca_features_no_classifier_ffnn],
        "D": [autoencoder_features_no_classifier_ffnn],
        "E": [computed_features_pca_combiner_ffnn],
        "F": [computed_features_autoencoder_combiner_ffnn],
        "G": [frequency_features_no_classifier_ffnn],
        "H": [frequency_entropy_features_ffnn],
        "I": [frequency_pca],
        "J": [frequency_autoencoder],
        "K": [frequency_embedded_pca],
        "L": [frequency_embedded_autoencoder],
        "M": [test_cnn_embedded]
    }
    hsc = None 
    # if none of the following porps are set FFNN is used
    use_svr = False
    use_gpr = False
    use_poly_reg = False
    do_eval(model_dict=models_dict, health_stage_classifier=hsc, use_svr=use_svr, use_gpr=use_gpr, use_poly_reg=use_poly_reg)

    models_dict = { 
        "A": [statistical_features_no_classifier_ffnn],
        "B": [entropy_features_no_classifier_ffnn],
        "C": [pca_features_no_classifier_ffnn],
        "D": [autoencoder_features_no_classifier_ffnn],
        "E": [computed_features_pca_combiner_ffnn],
        "F": [computed_features_autoencoder_combiner_ffnn],
        "G": [frequency_features_no_classifier_ffnn],
        "H": [frequency_entropy_features_ffnn],
        "I": [frequency_pca],
        "J": [frequency_autoencoder],
        "K": [frequency_embedded_pca],
        "L": [frequency_embedded_autoencoder],
        "M": [test_cnn_embedded]
    }
    hsc = LiEtAl2019HealthStageClassifier()
    # if none of the following porps are set FFNN is used
    use_svr = False
    use_gpr = True
    use_poly_reg = False
    do_eval(model_dict=models_dict, health_stage_classifier=hsc, use_svr=use_svr, use_gpr=use_gpr, use_poly_reg=use_poly_reg)

    models_dict = { 
        "A": [statistical_features_no_classifier_ffnn],
        "B": [entropy_features_no_classifier_ffnn],
        "C": [pca_features_no_classifier_ffnn],
        "D": [autoencoder_features_no_classifier_ffnn],
        "E": [computed_features_pca_combiner_ffnn],
        "F": [computed_features_autoencoder_combiner_ffnn],
        "G": [frequency_features_no_classifier_ffnn],
        "H": [frequency_entropy_features_ffnn],
        "I": [frequency_pca],
        "J": [frequency_autoencoder],
        "K": [frequency_embedded_pca],
        "L": [frequency_embedded_autoencoder],
        "M": [test_cnn_embedded]
    }
    hsc = None 
    # if none of the following porps are set FFNN is used
    use_svr = False
    use_gpr = True
    use_poly_reg = False
    do_eval(model_dict=models_dict, health_stage_classifier=hsc, use_svr=use_svr, use_gpr=use_gpr, use_poly_reg=use_poly_reg)
