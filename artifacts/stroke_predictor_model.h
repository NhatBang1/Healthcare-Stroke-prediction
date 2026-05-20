#ifndef STROKE_PREDICTOR_MODEL_H
#define STROKE_PREDICTOR_MODEL_H

#include <vector>
#include <string>
#include <cmath>
#include <algorithm>

namespace StrokeModel {

// Decision Threshold
const double DECISION_THRESHOLD = 0.5950;

// Feature Names (In Order)
const int NUM_FEATURES = 11;
const std::string FEATURE_NAMES[11] = {
    "age", "hypertension", "heart_disease", "avg_glucose_level", "bmi",
    "avg_RestingBP", "heart_cholesterol", "avg_MaxHR", "avg_Oldpeak",
    "heart_disease_rate", "Estimated_calories"
};

// Imputation Medians
struct ImputationMedians {
    double gender = 1.000000000;
    double age = 45.000000000;
    double hypertension = 0.000000000;
    double heart_disease = 0.000000000;
    double avg_glucose_level = 91.460000000;
    double bmi = 28.000000000;
    double avg_RestingBP = 131.500000000;
    double heart_cholesterol = 217.444444400;
    double avg_MaxHR = 141.096774200;
    double avg_Oldpeak = 0.814583333;
    double heart_disease_rate = 0.500000000;
};

const ImputationMedians MEDIANS;

// StandardScaler Parameters
const double SCALER_MEAN[11] = {
    43.143975398, 0.101481689, 0.049482807, 105.176142018, 28.760022365, 131.309765668, 217.963001074, 141.953729379, 0.790184017, 0.449162656, 98.946792848
};
const double SCALER_SCALE[11] = {
    22.716475498, 0.301965487, 0.216873831, 44.309682964, 7.490373716, 9.271830234, 42.815293029, 12.448945351, 0.419458169, 0.259266558, 17.816998927
};

// Logistic Regression Weights and Intercept
const double LR_COEFFICIENTS[11] = {
    1.297396822, 0.165980740, 0.071823247, 0.214127026, -0.206321329, -0.005946467, -0.088811419, 0.074341543, 0.089142237, -0.111926574, 0.509616878
};
const double LR_INTERCEPT = -1.124116438;

// XGBoost Calories Regressor Trees
inline double predict_tree_0(double Gender, double Age, double BMI) {
    if (BMI < 26.078971900) {
        if (BMI < 24.956596400) {
            if (Gender < 1.000000000) {
                if (BMI < 23.711845400) {
                    return -0.718623817;
                } else {
                    return -0.135565951;
                }
            } else {
                if (BMI < 24.515596400) {
                    return -0.089962937;
                } else {
                    return 0.169729501;
                }
            }
        } else {
            if (BMI < 25.057359700) {
                if (BMI < 24.989587800) {
                    return 0.063140363;
                } else {
                    return 0.636417031;
                }
            } else {
                if (BMI < 25.992439300) {
                    return 0.047143538;
                } else {
                    return -0.158739358;
                }
            }
        }
    } else {
        if (BMI < 27.173914000) {
            if (BMI < 26.312810900) {
                if (BMI < 26.282564200) {
                    return 0.455869555;
                } else {
                    return -0.093542211;
                }
            } else {
                if (BMI < 26.315052000) {
                    return 1.369385480;
                } else {
                    return 0.370667130;
                }
            }
        } else {
            if (BMI < 27.450605400) {
                return 1.021749020;
            } else {
                if (BMI < 27.472526600) {
                    return -0.017772522;
                } else {
                    return 0.637936234;
                }
            }
        }
    }
}

inline double predict_tree_1(double Gender, double Age, double BMI) {
    if (BMI < 26.078971900) {
        if (BMI < 24.956596400) {
            if (Gender < 1.000000000) {
                if (BMI < 23.423557300) {
                    return -0.821953118;
                } else {
                    return -0.142475322;
                }
            } else {
                if (BMI < 24.515596400) {
                    return -0.086104222;
                } else {
                    return 0.092072561;
                }
            }
        } else {
            if (BMI < 25.057359700) {
                if (BMI < 24.989587800) {
                    return 0.175126180;
                } else {
                    return 0.693853140;
                }
            } else {
                if (Gender < 1.000000000) {
                    return 0.051163573;
                } else {
                    return -0.115122832;
                }
            }
        }
    } else {
        if (BMI < 27.776929900) {
            if (BMI < 26.312810900) {
                if (BMI < 26.282564200) {
                    return 0.494183958;
                } else {
                    return -0.170679584;
                }
            } else {
                if (BMI < 26.315052000) {
                    return 1.278072600;
                } else {
                    return 0.386829674;
                }
            }
        } else {
            return 0.909285665;
        }
    }
}

inline double predict_tree_2(double Gender, double Age, double BMI) {
    if (Age < 49.000000000) {
        if (Age < 34.000000000) {
            if (BMI < 26.155626300) {
                if (BMI < 21.989893000) {
                    return -0.145413741;
                } else {
                    return -0.476528794;
                }
            } else {
                if (Age < 26.000000000) {
                    return -0.318888694;
                } else {
                    return 0.190558195;
                }
            }
        } else {
            if (BMI < 25.207756000) {
                if (BMI < 24.569315000) {
                    return -0.039846241;
                } else {
                    return -0.320020556;
                }
            } else {
                if (Age < 39.000000000) {
                    return -0.185625911;
                } else {
                    return 0.265729696;
                }
            }
        }
    } else {
        if (Age < 61.000000000) {
            if (BMI < 27.776929900) {
                if (Age < 60.000000000) {
                    return 0.224225864;
                } else {
                    return -0.240715474;
                }
            } else {
                if (Age < 53.000000000) {
                    return 2.393844130;
                } else {
                    return 0.841769338;
                }
            }
        } else {
            if (BMI < 26.037494700) {
                if (BMI < 22.758306500) {
                    return 0.087539464;
                } else {
                    return 0.564934552;
                }
            } else {
                if (BMI < 27.160493900) {
                    return 0.960900187;
                } else {
                    return 1.378598810;
                }
            }
        }
    }
}

inline double predict_tree_3(double Gender, double Age, double BMI) {
    if (Age < 49.000000000) {
        if (Age < 34.000000000) {
            if (BMI < 26.155626300) {
                if (BMI < 21.952478400) {
                    return -0.121964835;
                } else {
                    return -0.438243985;
                }
            } else {
                if (BMI < 26.827421200) {
                    return 0.154315442;
                } else {
                    return -0.512102365;
                }
            }
        } else {
            if (BMI < 25.207756000) {
                if (BMI < 21.096191400) {
                    return 0.653640032;
                } else {
                    return -0.124147519;
                }
            } else {
                if (Age < 41.000000000) {
                    return -0.123671301;
                } else {
                    return 0.363611758;
                }
            }
        }
    } else {
        if (Age < 61.000000000) {
            if (Age < 60.000000000) {
                if (Age < 57.000000000) {
                    return 0.167269558;
                } else {
                    return 0.505060554;
                }
            } else {
                if (BMI < 22.408786800) {
                    return -1.370186570;
                } else {
                    return -0.204627395;
                }
            }
        } else {
            if (BMI < 24.670375800) {
                if (BMI < 22.862533600) {
                    return -0.047509320;
                } else {
                    return 0.444725215;
                }
            } else {
                if (BMI < 26.555965400) {
                    return 0.678521574;
                } else {
                    return 1.069981340;
                }
            }
        }
    }
}

inline double predict_tree_4(double Gender, double Age, double BMI) {
    if (Age < 57.000000000) {
        if (Age < 34.000000000) {
            if (Age < 21.000000000) {
                if (Gender < 1.000000000) {
                    return -0.908617914;
                } else {
                    return -0.447551847;
                }
            } else {
                if (Gender < 1.000000000) {
                    return -0.392050654;
                } else {
                    return -0.270749897;
                }
            }
        } else {
            if (Age < 49.000000000) {
                if (Age < 48.000000000) {
                    return -0.019350661;
                } else {
                    return -0.420453429;
                }
            } else {
                if (Age < 56.000000000) {
                    return 0.180231720;
                } else {
                    return -0.311330169;
                }
            }
        }
    } else {
        if (Gender < 1.000000000) {
            if (Age < 61.000000000) {
                if (Age < 58.000000000) {
                    return 0.679952741;
                } else {
                    return 0.283209383;
                }
            } else {
                if (Age < 64.000000000) {
                    return 0.766564190;
                } else {
                    return 0.900570154;
                }
            }
        } else {
            if (Age < 73.000000000) {
                if (Age < 59.000000000) {
                    return 0.591097534;
                } else {
                    return 0.180977553;
                }
            } else {
                if (Age < 76.000000000) {
                    return 0.900468171;
                } else {
                    return 0.222434327;
                }
            }
        }
    }
}

inline double predict_tree_5(double Gender, double Age, double BMI) {
    if (Age < 57.000000000) {
        if (Age < 34.000000000) {
            if (Gender < 1.000000000) {
                if (Age < 21.000000000) {
                    return -0.820166767;
                } else {
                    return -0.401073396;
                }
            } else {
                if (Age < 26.000000000) {
                    return -0.321629167;
                } else {
                    return -0.224576622;
                }
            }
        } else {
            if (Age < 56.000000000) {
                if (Age < 49.000000000) {
                    return -0.023168387;
                } else {
                    return 0.179507136;
                }
            } else {
                if (Gender < 1.000000000) {
                    return -0.239330530;
                } else {
                    return -0.696855783;
                }
            }
        }
    } else {
        if (Gender < 1.000000000) {
            if (Age < 64.000000000) {
                if (Age < 62.000000000) {
                    return 0.567532241;
                } else {
                    return 0.410167962;
                }
            } else {
                if (Age < 66.000000000) {
                    return 1.175039770;
                } else {
                    return 0.808535457;
                }
            }
        } else {
            if (Age < 59.000000000) {
                return 0.759584785;
            } else {
                if (Age < 73.000000000) {
                    return 0.224054307;
                } else {
                    return 0.489884466;
                }
            }
        }
    }
}

inline double predict_tree_6(double Gender, double Age, double BMI) {
    if (Age < 57.000000000) {
        if (Age < 34.000000000) {
            if (Gender < 1.000000000) {
                if (Age < 21.000000000) {
                    return -0.792739868;
                } else {
                    return -0.399873108;
                }
            } else {
                if (Age < 21.000000000) {
                    return -0.362288803;
                } else {
                    return -0.223508775;
                }
            }
        } else {
            if (Age < 41.000000000) {
                if (Age < 40.000000000) {
                    return -0.072744496;
                } else {
                    return -0.237203017;
                }
            } else {
                if (Gender < 1.000000000) {
                    return 0.170452058;
                } else {
                    return -0.032677419;
                }
            }
        }
    } else {
        if (Gender < 1.000000000) {
            if (Age < 61.000000000) {
                if (Age < 58.000000000) {
                    return 0.822462976;
                } else {
                    return 0.332091480;
                }
            } else {
                if (Age < 62.000000000) {
                    return 1.316206460;
                } else {
                    return 0.816585839;
                }
            }
        } else {
            if (Age < 60.000000000) {
                if (Age < 58.000000000) {
                    return 0.204092950;
                } else {
                    return 0.545787632;
                }
            } else {
                if (Age < 63.000000000) {
                    return -0.127743185;
                } else {
                    return 0.286820799;
                }
            }
        }
    }
}

inline double predict_tree_7(double Gender, double Age, double BMI) {
    if (Age < 41.000000000) {
        if (Age < 34.000000000) {
            if (Age < 21.000000000) {
                if (Gender < 1.000000000) {
                    return -0.714471936;
                } else {
                    return -0.303742677;
                }
            } else {
                if (Age < 28.000000000) {
                    return -0.359157830;
                } else {
                    return -0.275191218;
                }
            }
        } else {
            if (Gender < 1.000000000) {
                if (Age < 36.000000000) {
                    return -0.223377511;
                } else {
                    return -0.106635146;
                }
            } else {
                if (Age < 40.000000000) {
                    return -0.027388346;
                } else {
                    return -0.244457766;
                }
            }
        }
    } else {
        if (Age < 57.000000000) {
            if (Age < 56.000000000) {
                if (Age < 49.000000000) {
                    return 0.047875918;
                } else {
                    return 0.237820521;
                }
            } else {
                if (Gender < 1.000000000) {
                    return -0.069873534;
                } else {
                    return -0.657349050;
                }
            }
        } else {
            if (Gender < 1.000000000) {
                if (Age < 64.000000000) {
                    return 0.502771080;
                } else {
                    return 0.780708671;
                }
            } else {
                if (Age < 73.000000000) {
                    return 0.201463193;
                } else {
                    return 0.405501008;
                }
            }
        }
    }
}

inline double predict_tree_8(double Gender, double Age, double BMI) {
    if (Age < 57.000000000) {
        if (Age < 34.000000000) {
            if (BMI < 21.718065300) {
                if (Age < 22.000000000) {
                    return -0.596212089;
                } else {
                    return 0.139042616;
                }
            } else {
                if (BMI < 26.196188000) {
                    return -0.378676981;
                } else {
                    return -0.046582095;
                }
            }
        } else {
            if (BMI < 24.956596400) {
                if (BMI < 24.560325600) {
                    return 0.007855396;
                } else {
                    return -0.386722952;
                }
            } else {
                if (BMI < 25.057359700) {
                    return 0.634586871;
                } else {
                    return 0.096527048;
                }
            }
        }
    } else {
        if (BMI < 24.670375800) {
            if (Age < 72.000000000) {
                if (BMI < 23.733238200) {
                    return 0.310778260;
                } else {
                    return -0.121950731;
                }
            } else {
                if (BMI < 22.790329000) {
                    return -0.103137612;
                } else {
                    return 0.684077680;
                }
            }
        } else {
            if (BMI < 26.037494700) {
                if (BMI < 26.014568300) {
                    return 0.565309107;
                } else {
                    return -0.562384009;
                }
            } else {
                if (Age < 61.000000000) {
                    return 0.523272455;
                } else {
                    return 0.870117426;
                }
            }
        }
    }
}

inline double predict_tree_9(double Gender, double Age, double BMI) {
    if (Age < 49.000000000) {
        if (Age < 32.000000000) {
            if (BMI < 21.989893000) {
                if (Age < 30.000000000) {
                    return 0.044432443;
                } else {
                    return -0.505445719;
                }
            } else {
                if (BMI < 26.040952700) {
                    return -0.373078346;
                } else {
                    return 0.011297299;
                }
            }
        } else {
            if (Age < 48.000000000) {
                if (BMI < 24.221452700) {
                    return -0.103322253;
                } else {
                    return 0.051867563;
                }
            } else {
                if (BMI < 26.869806300) {
                    return -0.546661377;
                } else {
                    return 1.182295560;
                }
            }
        }
    } else {
        if (BMI < 24.956596400) {
            if (Age < 72.000000000) {
                if (BMI < 24.913494100) {
                    return 0.189390332;
                } else {
                    return -0.526623070;
                }
            } else {
                if (BMI < 24.654832800) {
                    return 0.600534558;
                } else {
                    return -0.184366047;
                }
            }
        } else {
            if (Age < 61.000000000) {
                if (BMI < 25.249895100) {
                    return 0.679974914;
                } else {
                    return 0.232185379;
                }
            } else {
                if (BMI < 26.570304900) {
                    return 0.502285838;
                } else {
                    return 0.906776488;
                }
            }
        }
    }
}

inline double predict_tree_10(double Gender, double Age, double BMI) {
    if (Age < 47.000000000) {
        if (Age < 32.000000000) {
            if (Gender < 1.000000000) {
                if (Age < 21.000000000) {
                    return -0.764198899;
                } else {
                    return -0.362190545;
                }
            } else {
                if (Age < 22.000000000) {
                    return -0.309456497;
                } else {
                    return -0.199380472;
                }
            }
        } else {
            if (Age < 41.000000000) {
                if (Age < 40.000000000) {
                    return -0.058185697;
                } else {
                    return -0.148952678;
                }
            } else {
                if (Age < 46.000000000) {
                    return 0.066897549;
                } else {
                    return -0.145914033;
                }
            }
        }
    } else {
        if (Gender < 1.000000000) {
            if (Age < 61.000000000) {
                if (Age < 48.000000000) {
                    return 0.780890644;
                } else {
                    return 0.194482967;
                }
            } else {
                if (Age < 64.000000000) {
                    return 0.529992461;
                } else {
                    return 0.709333539;
                }
            }
        } else {
            if (Age < 73.000000000) {
                if (Age < 70.000000000) {
                    return 0.119196653;
                } else {
                    return -0.118451595;
                }
            } else {
                if (Age < 75.000000000) {
                    return 0.935974956;
                } else {
                    return 0.168287128;
                }
            }
        }
    }
}

inline double predict_tree_11(double Gender, double Age, double BMI) {
    if (BMI < 25.767219500) {
        if (BMI < 25.762678100) {
            if (BMI < 24.989587800) {
                if (Gender < 1.000000000) {
                    return -0.173493311;
                } else {
                    return -0.053251445;
                }
            } else {
                if (BMI < 25.057359700) {
                    return 0.462142944;
                } else {
                    return -0.027975716;
                }
            }
        } else {
            if (Gender < 1.000000000) {
                return -0.699011385;
            } else {
                return 0.493497580;
            }
        }
    } else {
        if (BMI < 27.137414900) {
            if (BMI < 25.795917500) {
                if (Gender < 1.000000000) {
                    return 0.465756059;
                } else {
                    return 1.195993540;
                }
            } else {
                if (BMI < 25.854639100) {
                    return -0.361273408;
                } else {
                    return 0.206111714;
                }
            }
        } else {
            if (BMI < 27.450605400) {
                if (BMI < 27.173914000) {
                    return 0.427876115;
                } else {
                    return 0.853016138;
                }
            } else {
                if (BMI < 27.472526600) {
                    return -0.044361617;
                } else {
                    return 0.481847674;
                }
            }
        }
    }
}

inline double predict_tree_12(double Gender, double Age, double BMI) {
    if (Age < 41.000000000) {
        if (Age < 32.000000000) {
            if (Gender < 1.000000000) {
                if (Age < 21.000000000) {
                    return -0.727176249;
                } else {
                    return -0.353482157;
                }
            } else {
                if (Age < 23.000000000) {
                    return -0.125607967;
                } else {
                    return -0.232321575;
                }
            }
        } else {
            if (Age < 35.000000000) {
                if (Age < 34.000000000) {
                    return -0.146849141;
                } else {
                    return -0.087741271;
                }
            } else {
                if (Age < 38.000000000) {
                    return 0.038212102;
                } else {
                    return -0.110857353;
                }
            }
        }
    } else {
        if (Age < 64.000000000) {
            if (Age < 57.000000000) {
                if (Age < 56.000000000) {
                    return 0.105554648;
                } else {
                    return -0.295369178;
                }
            } else {
                if (Age < 60.000000000) {
                    return 0.395254195;
                } else {
                    return 0.162439510;
                }
            }
        } else {
            if (Gender < 1.000000000) {
                if (Age < 66.000000000) {
                    return 1.108187560;
                } else {
                    return 0.614156008;
                }
            } else {
                if (Age < 76.000000000) {
                    return 0.330896497;
                } else {
                    return 0.063371390;
                }
            }
        }
    }
}

inline double predict_tree_13(double Gender, double Age, double BMI) {
    if (Age < 57.000000000) {
        if (Age < 34.000000000) {
            if (Age < 21.000000000) {
                if (Gender < 1.000000000) {
                    return -0.760219812;
                } else {
                    return -0.248500437;
                }
            } else {
                if (Gender < 1.000000000) {
                    return -0.303558052;
                } else {
                    return -0.190677121;
                }
            }
        } else {
            if (Age < 49.000000000) {
                if (Age < 48.000000000) {
                    return -0.012652603;
                } else {
                    return -0.321826994;
                }
            } else {
                if (Age < 56.000000000) {
                    return 0.139308676;
                } else {
                    return -0.230387792;
                }
            }
        }
    } else {
        if (Gender < 1.000000000) {
            if (Age < 61.000000000) {
                if (Age < 60.000000000) {
                    return 0.375038147;
                } else {
                    return 0.209113598;
                }
            } else {
                if (Age < 62.000000000) {
                    return 0.954810977;
                } else {
                    return 0.674043238;
                }
            }
        } else {
            if (Age < 59.000000000) {
                if (Age < 58.000000000) {
                    return 0.135936946;
                } else {
                    return 0.755707085;
                }
            } else {
                if (Age < 63.000000000) {
                    return -0.013900040;
                } else {
                    return 0.257205755;
                }
            }
        }
    }
}

inline double predict_tree_14(double Gender, double Age, double BMI) {
    if (Age < 49.000000000) {
        if (Age < 34.000000000) {
            if (BMI < 26.196188000) {
                if (BMI < 21.989893000) {
                    return -0.067030363;
                } else {
                    return -0.303534091;
                }
            } else {
                if (BMI < 26.827421200) {
                    return 0.260626137;
                } else {
                    return -0.342790604;
                }
            }
        } else {
            if (BMI < 21.096191400) {
                if (Age < 44.000000000) {
                    return 1.312239650;
                } else {
                    return 0.376151562;
                }
            } else {
                if (BMI < 25.207756000) {
                    return -0.104639269;
                } else {
                    return 0.109907769;
                }
            }
        }
    } else {
        if (Age < 64.000000000) {
            if (BMI < 26.023048400) {
                if (BMI < 25.992439300) {
                    return 0.138402551;
                } else {
                    return -0.976576746;
                }
            } else {
                if (BMI < 26.037494700) {
                    return 1.789119240;
                } else {
                    return 0.328356743;
                }
            }
        } else {
            if (BMI < 26.312810900) {
                if (BMI < 26.282564200) {
                    return 0.391302496;
                } else {
                    return -0.731851041;
                }
            } else {
                if (BMI < 26.869806300) {
                    return 0.947104335;
                } else {
                    return 0.452170491;
                }
            }
        }
    }
}

inline double predict_tree_15(double Gender, double Age, double BMI) {
    if (BMI < 25.767219500) {
        if (BMI < 24.654832800) {
            if (Gender < 1.000000000) {
                if (BMI < 23.711845400) {
                    return -0.499832898;
                } else {
                    return -0.083064899;
                }
            } else {
                if (BMI < 23.733238200) {
                    return -0.021044642;
                } else {
                    return -0.137244627;
                }
            }
        } else {
            if (BMI < 25.306932400) {
                if (BMI < 24.989587800) {
                    return -0.022248875;
                } else {
                    return 0.167958349;
                }
            } else {
                if (BMI < 25.381467800) {
                    return -0.452269197;
                } else {
                    return -0.062347930;
                }
            }
        }
    } else {
        if (BMI < 27.173914000) {
            if (BMI < 25.795917500) {
                if (Gender < 1.000000000) {
                    return 0.512543261;
                } else {
                    return 1.282409670;
                }
            } else {
                if (BMI < 25.854639100) {
                    return -0.519445181;
                } else {
                    return 0.166047528;
                }
            }
        } else {
            if (BMI < 27.379665400) {
                return 0.824736536;
            } else {
                if (BMI < 27.776929900) {
                    return 0.248143509;
                } else {
                    return 0.734192789;
                }
            }
        }
    }
}

inline double predict_tree_16(double Gender, double Age, double BMI) {
    if (Age < 49.000000000) {
        if (Age < 34.000000000) {
            if (Age < 21.000000000) {
                if (BMI < 20.504934300) {
                    return 0.754674792;
                } else {
                    return -0.519840539;
                }
            } else {
                if (BMI < 26.040952700) {
                    return -0.243817672;
                } else {
                    return -0.001317970;
                }
            }
        } else {
            if (BMI < 21.096191400) {
                if (Age < 44.000000000) {
                    return 1.164935230;
                } else {
                    return 0.364314556;
                }
            } else {
                if (Age < 48.000000000) {
                    return -0.036570326;
                } else {
                    return -0.401868284;
                }
            }
        }
    } else {
        if (BMI < 26.555965400) {
            if (Age < 57.000000000) {
                if (BMI < 26.423570600) {
                    return 0.137074217;
                } else {
                    return -0.828081667;
                }
            } else {
                if (BMI < 24.670375800) {
                    return 0.240973771;
                } else {
                    return 0.411974281;
                }
            }
        } else {
            if (Age < 73.000000000) {
                if (Age < 51.000000000) {
                    return 1.097900630;
                } else {
                    return 0.402302593;
                }
            } else {
                if (BMI < 26.583175700) {
                    return -0.162731662;
                } else {
                    return 1.089883450;
                }
            }
        }
    }
}

inline double predict_tree_17(double Gender, double Age, double BMI) {
    if (BMI < 25.767219500) {
        if (BMI < 21.773841900) {
            if (BMI < 21.545091600) {
                if (BMI < 21.367521300) {
                    return 0.062577657;
                } else {
                    return -0.200039700;
                }
            } else {
                if (BMI < 21.612812000) {
                    return 0.496807516;
                } else {
                    return 0.227782175;
                }
            }
        } else {
            if (BMI < 22.282478300) {
                if (BMI < 21.952478400) {
                    return -0.056142330;
                } else {
                    return -0.381569505;
                }
            } else {
                if (BMI < 25.306932400) {
                    return -0.021838883;
                } else {
                    return -0.138800547;
                }
            }
        }
    } else {
        if (BMI < 26.570304900) {
            if (BMI < 26.423570600) {
                if (BMI < 26.312810900) {
                    return 0.135986701;
                } else {
                    return 0.579361439;
                }
            } else {
                if (BMI < 26.528511000) {
                    return -0.295707941;
                } else {
                    return 0.088593014;
                }
            }
        } else {
            if (Gender < 1.000000000) {
                if (BMI < 27.776929900) {
                    return 0.255059361;
                } else {
                    return 0.605728686;
                }
            } else {
                return 0.931389272;
            }
        }
    }
}

inline double predict_tree_18(double Gender, double Age, double BMI) {
    if (Age < 41.000000000) {
        if (Age < 21.000000000) {
            if (Gender < 1.000000000) {
                return -0.747986197;
            } else {
                return -0.320156366;
            }
        } else {
            if (Age < 32.000000000) {
                if (Gender < 1.000000000) {
                    return -0.290744841;
                } else {
                    return -0.154416993;
                }
            } else {
                if (Gender < 1.000000000) {
                    return -0.107456595;
                } else {
                    return -0.026414428;
                }
            }
        }
    } else {
        if (Age < 57.000000000) {
            if (Age < 42.000000000) {
                if (Gender < 1.000000000) {
                    return 0.198185265;
                } else {
                    return 0.494031549;
                }
            } else {
                if (Gender < 1.000000000) {
                    return 0.126503944;
                } else {
                    return -0.072203629;
                }
            }
        } else {
            if (Gender < 1.000000000) {
                if (Age < 64.000000000) {
                    return 0.279114217;
                } else {
                    return 0.589764297;
                }
            } else {
                if (Age < 60.000000000) {
                    return 0.381647229;
                } else {
                    return 0.091611333;
                }
            }
        }
    }
}

inline double predict_tree_19(double Gender, double Age, double BMI) {
    if (Age < 47.000000000) {
        if (Age < 32.000000000) {
            if (Gender < 1.000000000) {
                if (Age < 21.000000000) {
                    return -0.576357126;
                } else {
                    return -0.271576881;
                }
            } else {
                if (Age < 29.000000000) {
                    return -0.177375242;
                } else {
                    return -0.063911043;
                }
            }
        } else {
            if (Age < 46.000000000) {
                if (Age < 41.000000000) {
                    return -0.039620485;
                } else {
                    return 0.027124442;
                }
            } else {
                return -0.196177378;
            }
        }
    } else {
        if (Gender < 1.000000000) {
            if (Age < 64.000000000) {
                if (Age < 48.000000000) {
                    return 0.699655056;
                } else {
                    return 0.232946754;
                }
            } else {
                if (Age < 77.000000000) {
                    return 0.553548515;
                } else {
                    return 0.822101295;
                }
            }
        } else {
            if (Age < 73.000000000) {
                if (Age < 70.000000000) {
                    return 0.069113783;
                } else {
                    return -0.235250160;
                }
            } else {
                if (Age < 75.000000000) {
                    return 0.698029816;
                } else {
                    return 0.094289988;
                }
            }
        }
    }
}

inline double predict_tree_20(double Gender, double Age, double BMI) {
    if (BMI < 26.196188000) {
        if (BMI < 24.772096600) {
            if (BMI < 21.218317000) {
                if (BMI < 20.796730000) {
                    return -0.123080455;
                } else {
                    return 0.377446622;
                }
            } else {
                if (BMI < 22.230987500) {
                    return -0.173111305;
                } else {
                    return -0.044752721;
                }
            }
        } else {
            if (BMI < 25.306932400) {
                if (BMI < 24.967113500) {
                    return 0.013753339;
                } else {
                    return 0.191269889;
                }
            } else {
                if (BMI < 25.333334000) {
                    return -0.807196856;
                } else {
                    return 0.025639979;
                }
            }
        }
    } else {
        if (BMI < 26.264944100) {
            if (BMI < 26.234567600) {
                if (Gender < 1.000000000) {
                    return 0.484767705;
                } else {
                    return -0.118606001;
                }
            } else {
                return 1.186209440;
            }
        } else {
            if (BMI < 27.776929900) {
                if (BMI < 27.450605400) {
                    return 0.194241405;
                } else {
                    return -0.104566827;
                }
            } else {
                return 0.731098056;
            }
        }
    }
}

inline double predict_tree_21(double Gender, double Age, double BMI) {
    if (BMI < 25.767219500) {
        if (BMI < 21.773841900) {
            if (BMI < 21.545091600) {
                if (BMI < 21.218317000) {
                    return 0.199416980;
                } else {
                    return -0.097189099;
                }
            } else {
                if (BMI < 21.612812000) {
                    return 0.534131825;
                } else {
                    return 0.133359462;
                }
            }
        } else {
            if (BMI < 22.100290300) {
                if (BMI < 21.989893000) {
                    return -0.250362515;
                } else {
                    return -0.584056556;
                }
            } else {
                if (BMI < 24.772096600) {
                    return -0.078205362;
                } else {
                    return 0.006216488;
                }
            }
        }
    } else {
        if (BMI < 26.851852400) {
            if (BMI < 25.795917500) {
                if (Gender < 1.000000000) {
                    return 0.496239990;
                } else {
                    return 1.562815790;
                }
            } else {
                if (BMI < 25.854639100) {
                    return -0.549603522;
                } else {
                    return 0.110710941;
                }
            }
        } else {
            if (BMI < 27.450605400) {
                if (BMI < 27.173914000) {
                    return 0.321048945;
                } else {
                    return 0.723754406;
                }
            } else {
                if (BMI < 27.776929900) {
                    return 0.023569949;
                } else {
                    return 0.522753835;
                }
            }
        }
    }
}

inline double predict_tree_22(double Gender, double Age, double BMI) {
    if (BMI < 26.312810900) {
        if (BMI < 24.956596400) {
            if (Gender < 1.000000000) {
                if (BMI < 23.711845400) {
                    return -0.549647331;
                } else {
                    return -0.078126274;
                }
            } else {
                if (BMI < 21.829952200) {
                    return 0.112325497;
                } else {
                    return -0.023895379;
                }
            }
        } else {
            if (BMI < 25.057359700) {
                if (BMI < 24.989587800) {
                    return 0.026924647;
                } else {
                    return 0.598502159;
                }
            } else {
                if (BMI < 26.303619400) {
                    return 0.015536773;
                } else {
                    return -0.545216382;
                }
            }
        }
    } else {
        if (BMI < 26.315052000) {
            if (Gender < 1.000000000) {
                return 1.184428330;
            } else {
                return 0.149676353;
            }
        } else {
            if (BMI < 26.528511000) {
                if (BMI < 26.423570600) {
                    return 0.245806053;
                } else {
                    return -0.284744382;
                }
            } else {
                if (Gender < 1.000000000) {
                    return 0.276081294;
                } else {
                    return 1.297855850;
                }
            }
        }
    }
}

inline double predict_tree_23(double Gender, double Age, double BMI) {
    if (Age < 57.000000000) {
        if (Age < 32.000000000) {
            if (Gender < 1.000000000) {
                if (Age < 21.000000000) {
                    return -0.635992467;
                } else {
                    return -0.247880965;
                }
            } else {
                if (Age < 29.000000000) {
                    return -0.173609540;
                } else {
                    return -0.036282137;
                }
            }
        } else {
            if (Gender < 1.000000000) {
                if (Age < 39.000000000) {
                    return -0.097258672;
                } else {
                    return 0.116057001;
                }
            } else {
                if (Age < 56.000000000) {
                    return -0.041367851;
                } else {
                    return -0.555729449;
                }
            }
        }
    } else {
        if (Gender < 1.000000000) {
            if (Age < 61.000000000) {
                if (Age < 58.000000000) {
                    return 0.666447103;
                } else {
                    return 0.178090513;
                }
            } else {
                if (Age < 62.000000000) {
                    return 0.934518278;
                } else {
                    return 0.493776470;
                }
            }
        } else {
            if (Age < 59.000000000) {
                if (Age < 58.000000000) {
                    return 0.272658378;
                } else {
                    return 0.532842398;
                }
            } else {
                if (Age < 76.000000000) {
                    return 0.144896790;
                } else {
                    return -0.082342126;
                }
            }
        }
    }
}

inline double predict_tree_24(double Gender, double Age, double BMI) {
    if (Age < 47.000000000) {
        if (Age < 32.000000000) {
            if (BMI < 21.952478400) {
                if (Age < 22.000000000) {
                    return -0.297085553;
                } else {
                    return 0.135799378;
                }
            } else {
                if (BMI < 26.040952700) {
                    return -0.277536571;
                } else {
                    return 0.055546135;
                }
            }
        } else {
            if (BMI < 26.570304900) {
                if (BMI < 25.661151900) {
                    return -0.048642911;
                } else {
                    return 0.148546457;
                }
            } else {
                if (Age < 39.000000000) {
                    return -0.706782162;
                } else {
                    return 0.070553623;
                }
            }
        }
    } else {
        if (Age < 57.000000000) {
            if (BMI < 21.773841900) {
                if (Age < 56.000000000) {
                    return 0.957937598;
                } else {
                    return -0.094439961;
                }
            } else {
                if (BMI < 22.230987500) {
                    return -0.558903813;
                } else {
                    return 0.096855499;
                }
            }
        } else {
            if (BMI < 24.670375800) {
                if (Age < 72.000000000) {
                    return 0.129518047;
                } else {
                    return 0.472473145;
                }
            } else {
                if (BMI < 24.858366000) {
                    return 0.838475108;
                } else {
                    return 0.394405246;
                }
            }
        }
    }
}

inline double predict_tree_25(double Gender, double Age, double BMI) {
    if (BMI < 26.196188000) {
        if (BMI < 21.218317000) {
            if (BMI < 20.957170500) {
                if (BMI < 20.504934300) {
                    return 0.338902235;
                } else {
                    return 0.078089938;
                }
            } else {
                if (BMI < 21.096191400) {
                    return 0.396149099;
                } else {
                    return 0.166223153;
                }
            }
        } else {
            if (BMI < 24.956596400) {
                if (Gender < 1.000000000) {
                    return -0.149966374;
                } else {
                    return -0.028610047;
                }
            } else {
                if (BMI < 25.057359700) {
                    return 0.397227138;
                } else {
                    return -0.040468540;
                }
            }
        }
    } else {
        if (BMI < 26.282564200) {
            if (Gender < 1.000000000) {
                if (BMI < 26.264944100) {
                    return 0.524579287;
                } else {
                    return 0.298842132;
                }
            } else {
                return -0.713734329;
            }
        } else {
            if (BMI < 26.296566000) {
                if (Gender < 1.000000000) {
                    return -0.447339118;
                } else {
                    return 0.004884745;
                }
            } else {
                if (BMI < 26.851852400) {
                    return 0.093071662;
                } else {
                    return 0.269308329;
                }
            }
        }
    }
}

inline double predict_tree_26(double Gender, double Age, double BMI) {
    if (BMI < 26.312810900) {
        if (BMI < 24.956596400) {
            if (BMI < 24.930748000) {
                if (BMI < 21.773841900) {
                    return 0.109023064;
                } else {
                    return -0.047111671;
                }
            } else {
                if (Gender < 1.000000000) {
                    return -0.678408265;
                } else {
                    return -0.000611945;
                }
            }
        } else {
            if (BMI < 25.057359700) {
                if (BMI < 24.989587800) {
                    return 0.072940819;
                } else {
                    return 0.583886564;
                }
            } else {
                if (BMI < 26.282564200) {
                    return 0.050177548;
                } else {
                    return -0.321493447;
                }
            }
        }
    } else {
        if (BMI < 26.315052000) {
            if (Gender < 1.000000000) {
                return 1.267212750;
            } else {
                return -0.204989925;
            }
        } else {
            if (BMI < 27.450605400) {
                if (BMI < 27.173914000) {
                    return 0.224339619;
                } else {
                    return 0.758635938;
                }
            } else {
                if (BMI < 27.776929900) {
                    return -0.104143292;
                } else {
                    return 0.306002796;
                }
            }
        }
    }
}

inline double predict_tree_27(double Gender, double Age, double BMI) {
    if (Age < 49.000000000) {
        if (Age < 28.000000000) {
            if (BMI < 22.837369900) {
                if (BMI < 22.714681600) {
                    return -0.107519388;
                } else {
                    return 0.545342922;
                }
            } else {
                if (BMI < 22.991689700) {
                    return -0.766202748;
                } else {
                    return -0.255954117;
                }
            }
        } else {
            if (Age < 48.000000000) {
                if (Age < 47.000000000) {
                    return -0.048677135;
                } else {
                    return 0.343098551;
                }
            } else {
                if (BMI < 26.851852400) {
                    return -0.442205995;
                } else {
                    return 1.047610160;
                }
            }
        }
    } else {
        if (Age < 61.000000000) {
            if (BMI < 27.776929900) {
                if (BMI < 27.173914000) {
                    return 0.110782385;
                } else {
                    return -0.435548395;
                }
            } else {
                if (Age < 53.000000000) {
                    return 2.132752660;
                } else {
                    return 0.675000966;
                }
            }
        } else {
            if (BMI < 23.046875000) {
                if (BMI < 22.145328500) {
                    return 0.616022170;
                } else {
                    return -0.201033562;
                }
            } else {
                if (BMI < 23.147254900) {
                    return 1.388177630;
                } else {
                    return 0.320346951;
                }
            }
        }
    }
}

inline double predict_tree_28(double Gender, double Age, double BMI) {
    if (BMI < 24.772096600) {
        if (BMI < 24.560325600) {
            if (BMI < 24.515596400) {
                if (BMI < 21.877550100) {
                    return 0.087722965;
                } else {
                    return -0.044490796;
                }
            } else {
                if (Gender < 1.000000000) {
                    return -0.125325933;
                } else {
                    return 0.636793554;
                }
            }
        } else {
            if (BMI < 24.618103000) {
                if (BMI < 24.609375000) {
                    return -0.204613015;
                } else {
                    return -1.309600590;
                }
            } else {
                if (BMI < 24.740938200) {
                    return 0.040371813;
                } else {
                    return -0.358607411;
                }
            }
        }
    } else {
        if (BMI < 24.835645700) {
            if (Gender < 1.000000000) {
                if (BMI < 24.810964600) {
                    return -0.068622380;
                } else {
                    return 0.545254230;
                }
            } else {
                if (BMI < 24.784257900) {
                    return 0.355923504;
                } else {
                    return 0.935299456;
                }
            }
        } else {
            if (Gender < 1.000000000) {
                if (BMI < 24.967113500) {
                    return -0.167550787;
                } else {
                    return 0.106871776;
                }
            } else {
                if (BMI < 25.181078000) {
                    return 0.074200988;
                } else {
                    return -0.228298828;
                }
            }
        }
    }
}

inline double predict_tree_29(double Gender, double Age, double BMI) {
    if (Age < 49.000000000) {
        if (Age < 34.000000000) {
            if (Gender < 1.000000000) {
                if (Age < 21.000000000) {
                    return -0.500246763;
                } else {
                    return -0.227134094;
                }
            } else {
                if (Age < 31.000000000) {
                    return -0.110026568;
                } else {
                    return -0.212795526;
                }
            }
        } else {
            if (Age < 48.000000000) {
                if (Age < 47.000000000) {
                    return -0.009463027;
                } else {
                    return 0.221909449;
                }
            } else {
                if (Gender < 1.000000000) {
                    return -0.299710810;
                } else {
                    return -0.075353205;
                }
            }
        }
    } else {
        if (Gender < 1.000000000) {
            if (Age < 61.000000000) {
                if (Age < 51.000000000) {
                    return 0.408474773;
                } else {
                    return 0.141146168;
                }
            } else {
                if (Age < 70.000000000) {
                    return 0.390650958;
                } else {
                    return 0.553696334;
                }
            }
        } else {
            if (Age < 63.000000000) {
                if (Age < 60.000000000) {
                    return 0.071436092;
                } else {
                    return -0.141871363;
                }
            } else {
                if (Age < 70.000000000) {
                    return 0.296533197;
                } else {
                    return 0.144694671;
                }
            }
        }
    }
}

inline double predict_tree_30(double Gender, double Age, double BMI) {
    if (Age < 57.000000000) {
        if (Age < 35.000000000) {
            if (Gender < 1.000000000) {
                if (Age < 21.000000000) {
                    return -0.480078697;
                } else {
                    return -0.192112714;
                }
            } else {
                if (Age < 33.000000000) {
                    return -0.113346346;
                } else {
                    return -0.033856787;
                }
            }
        } else {
            if (Age < 49.000000000) {
                if (Age < 48.000000000) {
                    return -0.011414515;
                } else {
                    return -0.389850885;
                }
            } else {
                if (Age < 56.000000000) {
                    return 0.117309131;
                } else {
                    return -0.268604994;
                }
            }
        }
    } else {
        if (Gender < 1.000000000) {
            if (Age < 58.000000000) {
                return 0.738240480;
            } else {
                if (Age < 60.000000000) {
                    return 0.145148262;
                } else {
                    return 0.445573688;
                }
            }
        } else {
            if (Age < 59.000000000) {
                if (Age < 58.000000000) {
                    return 0.262828231;
                } else {
                    return 0.702329516;
                }
            } else {
                if (Age < 63.000000000) {
                    return -0.057666812;
                } else {
                    return 0.189744890;
                }
            }
        }
    }
}

inline double predict_tree_31(double Gender, double Age, double BMI) {
    if (Age < 47.000000000) {
        if (Age < 29.000000000) {
            if (BMI < 26.880950900) {
                if (BMI < 21.718065300) {
                    return 0.040489070;
                } else {
                    return -0.243345276;
                }
            } else {
                if (Age < 27.000000000) {
                    return -0.207569271;
                } else {
                    return 1.608656410;
                }
            }
        } else {
            if (BMI < 26.570304900) {
                if (BMI < 24.242424000) {
                    return -0.079335652;
                } else {
                    return 0.029177602;
                }
            } else {
                if (Age < 39.000000000) {
                    return -0.694916606;
                } else {
                    return -0.081237897;
                }
            }
        }
    } else {
        if (BMI < 26.078971900) {
            if (BMI < 21.773841900) {
                if (BMI < 21.718065300) {
                    return 0.413735598;
                } else {
                    return 1.812301750;
                }
            } else {
                if (BMI < 21.913805000) {
                    return -1.198029160;
                } else {
                    return 0.132563233;
                }
            }
        } else {
            if (Age < 51.000000000) {
                if (BMI < 26.196188000) {
                    return 1.726461770;
                } else {
                    return 0.620326459;
                }
            } else {
                if (Age < 54.000000000) {
                    return -0.493822247;
                } else {
                    return 0.411482036;
                }
            }
        }
    }
}

inline double predict_tree_32(double Gender, double Age, double BMI) {
    if (Age < 34.000000000) {
        if (Age < 21.000000000) {
            if (BMI < 20.504934300) {
                return 1.003256560;
            } else {
                if (BMI < 22.282478300) {
                    return -0.668463647;
                } else {
                    return -0.349538177;
                }
            }
        } else {
            if (BMI < 21.877550100) {
                if (Age < 22.000000000) {
                    return -0.690131009;
                } else {
                    return 0.179604143;
                }
            } else {
                if (Age < 28.000000000) {
                    return -0.213601112;
                } else {
                    return -0.083262756;
                }
            }
        }
    } else {
        if (Age < 57.000000000) {
            if (BMI < 26.423570600) {
                if (BMI < 25.333334000) {
                    return 0.012809790;
                } else {
                    return 0.193001166;
                }
            } else {
                if (Age < 43.000000000) {
                    return -0.593027651;
                } else {
                    return 0.014375850;
                }
            }
        } else {
            if (BMI < 26.037494700) {
                if (BMI < 26.014568300) {
                    return 0.185553163;
                } else {
                    return -0.508685648;
                }
            } else {
                if (Age < 74.000000000) {
                    return 0.475303590;
                } else {
                    return 0.179582059;
                }
            }
        }
    }
}

inline double predict_tree_33(double Gender, double Age, double BMI) {
    if (Age < 57.000000000) {
        if (Age < 34.000000000) {
            if (Gender < 1.000000000) {
                if (Age < 21.000000000) {
                    return -0.445638835;
                } else {
                    return -0.183869243;
                }
            } else {
                if (Age < 23.000000000) {
                    return -0.059342537;
                } else {
                    return -0.140211031;
                }
            }
        } else {
            if (Age < 56.000000000) {
                if (Age < 49.000000000) {
                    return -0.029010648;
                } else {
                    return 0.065159313;
                }
            } else {
                if (Gender < 1.000000000) {
                    return 0.093901940;
                } else {
                    return -0.523228765;
                }
            }
        }
    } else {
        if (Gender < 1.000000000) {
            if (Age < 77.000000000) {
                if (Age < 74.000000000) {
                    return 0.336406648;
                } else {
                    return 0.066954158;
                }
            } else {
                if (Age < 78.000000000) {
                    return 0.701700807;
                } else {
                    return 0.464441389;
                }
            }
        } else {
            if (Age < 59.000000000) {
                if (Age < 58.000000000) {
                    return 0.244461745;
                } else {
                    return 0.375168651;
                }
            } else {
                if (Age < 62.000000000) {
                    return -0.209425375;
                } else {
                    return 0.125906423;
                }
            }
        }
    }
}

inline double predict_tree_34(double Gender, double Age, double BMI) {
    if (BMI < 26.312810900) {
        if (BMI < 26.282564200) {
            if (BMI < 25.880571400) {
                if (BMI < 25.795917500) {
                    return -0.026289156;
                } else {
                    return -0.427274764;
                }
            } else {
                if (BMI < 25.992439300) {
                    return 0.339982420;
                } else {
                    return 0.040486179;
                }
            }
        } else {
            if (BMI < 26.303619400) {
                if (BMI < 26.296566000) {
                    return -0.434838474;
                } else {
                    return 0.076572560;
                }
            } else {
                return -0.482866168;
            }
        }
    } else {
        if (BMI < 26.315052000) {
            if (Gender < 1.000000000) {
                return 0.993097961;
            } else {
                return -0.211908072;
            }
        } else {
            if (BMI < 26.583175700) {
                if (BMI < 26.423570600) {
                    return 0.116625905;
                } else {
                    return -0.056542229;
                }
            } else {
                if (BMI < 27.450605400) {
                    return 0.238508403;
                } else {
                    return 0.032187525;
                }
            }
        }
    }
}

inline double predict_tree_35(double Gender, double Age, double BMI) {
    if (Age < 57.000000000) {
        if (Age < 32.000000000) {
            if (Gender < 1.000000000) {
                if (Age < 21.000000000) {
                    return -0.439682156;
                } else {
                    return -0.198346898;
                }
            } else {
                if (Age < 26.000000000) {
                    return -0.141766161;
                } else {
                    return -0.048228495;
                }
            }
        } else {
            if (Age < 56.000000000) {
                if (Age < 54.000000000) {
                    return -0.005228321;
                } else {
                    return 0.209620506;
                }
            } else {
                if (Gender < 1.000000000) {
                    return 0.009684230;
                } else {
                    return -0.567836344;
                }
            }
        }
    } else {
        if (Gender < 1.000000000) {
            if (Age < 61.000000000) {
                if (Age < 58.000000000) {
                    return 0.477647871;
                } else {
                    return 0.143203929;
                }
            } else {
                if (Age < 62.000000000) {
                    return 0.835374713;
                } else {
                    return 0.397737950;
                }
            }
        } else {
            if (Age < 76.000000000) {
                if (Age < 73.000000000) {
                    return 0.074250653;
                } else {
                    return 0.467727154;
                }
            } else {
                if (Age < 77.000000000) {
                    return -0.359052986;
                } else {
                    return 0.004544694;
                }
            }
        }
    }
}

inline double predict_tree_36(double Gender, double Age, double BMI) {
    if (Age < 49.000000000) {
        if (Age < 35.000000000) {
            if (Gender < 1.000000000) {
                if (Age < 21.000000000) {
                    return -0.496677011;
                } else {
                    return -0.180877820;
                }
            } else {
                if (Age < 31.000000000) {
                    return -0.046631549;
                } else {
                    return -0.150457785;
                }
            }
        } else {
            if (Age < 48.000000000) {
                if (Age < 47.000000000) {
                    return -0.004696060;
                } else {
                    return 0.304946691;
                }
            } else {
                return -0.296054780;
            }
        }
    } else {
        if (Gender < 1.000000000) {
            if (Age < 77.000000000) {
                if (Age < 51.000000000) {
                    return 0.496443331;
                } else {
                    return 0.210083455;
                }
            } else {
                if (Age < 78.000000000) {
                    return 0.877861857;
                } else {
                    return 0.502704084;
                }
            }
        } else {
            if (Age < 73.000000000) {
                if (Age < 70.000000000) {
                    return 0.097020693;
                } else {
                    return -0.227297544;
                }
            } else {
                if (Age < 75.000000000) {
                    return 0.631951988;
                } else {
                    return 0.034091968;
                }
            }
        }
    }
}

inline double predict_tree_37(double Gender, double Age, double BMI) {
    if (Age < 57.000000000) {
        if (Age < 41.000000000) {
            if (BMI < 21.989893000) {
                if (Age < 22.000000000) {
                    return -0.304267675;
                } else {
                    return 0.146257237;
                }
            } else {
                if (BMI < 22.230987500) {
                    return -0.515538871;
                } else {
                    return -0.142695904;
                }
            }
        } else {
            if (BMI < 24.976087600) {
                if (BMI < 24.538965200) {
                    return 0.044620354;
                } else {
                    return -0.402126193;
                }
            } else {
                if (BMI < 25.977846100) {
                    return 0.256738365;
                } else {
                    return -0.051478576;
                }
            }
        }
    } else {
        if (BMI < 26.555965400) {
            if (BMI < 26.485540400) {
                if (BMI < 26.037494700) {
                    return 0.165136397;
                } else {
                    return 0.398737550;
                }
            } else {
                if (Age < 60.000000000) {
                    return 0.647675157;
                } else {
                    return -1.093456750;
                }
            }
        } else {
            if (BMI < 26.880950900) {
                if (Age < 79.000000000) {
                    return 0.743336797;
                } else {
                    return -0.722354472;
                }
            } else {
                if (Age < 77.000000000) {
                    return 0.135321364;
                } else {
                    return 0.966986597;
                }
            }
        }
    }
}

inline double predict_tree_38(double Gender, double Age, double BMI) {
    if (Age < 57.000000000) {
        if (Age < 34.000000000) {
            if (Age < 22.000000000) {
                if (Gender < 1.000000000) {
                    return -0.345850945;
                } else {
                    return -0.128177494;
                }
            } else {
                if (Age < 23.000000000) {
                    return -0.033826146;
                } else {
                    return -0.127901092;
                }
            }
        } else {
            if (Age < 56.000000000) {
                if (Age < 49.000000000) {
                    return 0.014211485;
                } else {
                    return 0.101306438;
                }
            } else {
                if (Gender < 1.000000000) {
                    return -0.033195853;
                } else {
                    return -0.508769274;
                }
            }
        }
    } else {
        if (Gender < 1.000000000) {
            if (Age < 61.000000000) {
                if (Age < 58.000000000) {
                    return 0.459110022;
                } else {
                    return 0.113752075;
                }
            } else {
                if (Age < 62.000000000) {
                    return 0.859477699;
                } else {
                    return 0.359698415;
                }
            }
        } else {
            if (Age < 59.000000000) {
                if (Age < 58.000000000) {
                    return 0.217658177;
                } else {
                    return 0.513393700;
                }
            } else {
                if (Age < 73.000000000) {
                    return 0.014717234;
                } else {
                    return 0.232377335;
                }
            }
        }
    }
}

inline double predict_tree_39(double Gender, double Age, double BMI) {
    if (Age < 47.000000000) {
        if (Age < 34.000000000) {
            if (Gender < 1.000000000) {
                if (Age < 31.000000000) {
                    return -0.237774059;
                } else {
                    return -0.090174958;
                }
            } else {
                if (Age < 31.000000000) {
                    return -0.068672568;
                } else {
                    return -0.238172293;
                }
            }
        } else {
            if (Age < 36.000000000) {
                if (Gender < 1.000000000) {
                    return -0.112714946;
                } else {
                    return 0.170490459;
                }
            } else {
                if (Age < 37.000000000) {
                    return -0.281188965;
                } else {
                    return -0.069693938;
                }
            }
        }
    } else {
        if (Gender < 1.000000000) {
            if (Age < 60.000000000) {
                if (Age < 48.000000000) {
                    return 0.725593984;
                } else {
                    return 0.059227154;
                }
            } else {
                if (Age < 62.000000000) {
                    return 0.619224429;
                } else {
                    return 0.373263359;
                }
            }
        } else {
            if (Age < 63.000000000) {
                if (Age < 60.000000000) {
                    return 0.035806336;
                } else {
                    return -0.188153580;
                }
            } else {
                if (Age < 67.000000000) {
                    return 0.280009270;
                } else {
                    return 0.069687106;
                }
            }
        }
    }
}

inline double predict_tree_40(double Gender, double Age, double BMI) {
    if (Age < 57.000000000) {
        if (Age < 22.000000000) {
            if (Gender < 1.000000000) {
                if (Age < 21.000000000) {
                    return -0.423656762;
                } else {
                    return -0.260009468;
                }
            } else {
                if (Age < 21.000000000) {
                    return -0.192445844;
                } else {
                    return -0.127889588;
                }
            }
        } else {
            if (Age < 56.000000000) {
                if (Age < 49.000000000) {
                    return -0.039706230;
                } else {
                    return 0.099481918;
                }
            } else {
                if (Gender < 1.000000000) {
                    return -0.132693708;
                } else {
                    return -0.493483871;
                }
            }
        }
    } else {
        if (Gender < 1.000000000) {
            if (Age < 75.000000000) {
                if (Age < 64.000000000) {
                    return 0.169346541;
                } else {
                    return 0.317171425;
                }
            } else {
                if (Age < 77.000000000) {
                    return 0.409268320;
                } else {
                    return 0.614424527;
                }
            }
        } else {
            if (Age < 60.000000000) {
                if (Age < 59.000000000) {
                    return 0.357962877;
                } else {
                    return 0.252374381;
                }
            } else {
                if (Age < 62.000000000) {
                    return -0.242084309;
                } else {
                    return 0.107630171;
                }
            }
        }
    }
}

inline double predict_tree_41(double Gender, double Age, double BMI) {
    if (Age < 57.000000000) {
        if (Age < 34.000000000) {
            if (BMI < 22.862533600) {
                if (BMI < 22.829963700) {
                    return -0.069137931;
                } else {
                    return 0.347569764;
                }
            } else {
                if (BMI < 22.991689700) {
                    return -0.765971899;
                } else {
                    return -0.133977145;
                }
            }
        } else {
            if (BMI < 21.096191400) {
                if (Age < 51.000000000) {
                    return 0.703408957;
                } else {
                    return -1.167918800;
                }
            } else {
                if (BMI < 22.230987500) {
                    return -0.228018090;
                } else {
                    return 0.020475293;
                }
            }
        }
    } else {
        if (BMI < 27.472526600) {
            if (BMI < 27.173914000) {
                if (Age < 76.000000000) {
                    return 0.196489230;
                } else {
                    return 0.000239720;
                }
            } else {
                if (Age < 63.000000000) {
                    return -0.142252013;
                } else {
                    return 0.995304823;
                }
            }
        } else {
            if (Age < 60.000000000) {
                if (Age < 58.000000000) {
                    return -0.079474993;
                } else {
                    return 1.442144750;
                }
            } else {
                if (Age < 64.000000000) {
                    return -1.481407400;
                } else {
                    return -0.038561039;
                }
            }
        }
    }
}

inline double predict_tree_42(double Gender, double Age, double BMI) {
    if (BMI < 26.078971900) {
        if (BMI < 25.992439300) {
            if (BMI < 25.767219500) {
                if (BMI < 25.306932400) {
                    return -0.001906455;
                } else {
                    return -0.132095754;
                }
            } else {
                if (BMI < 25.795917500) {
                    return 0.600098073;
                } else {
                    return 0.070155188;
                }
            }
        } else {
            if (BMI < 26.023048400) {
                if (BMI < 26.014568300) {
                    return -0.109889880;
                } else {
                    return -1.021700980;
                }
            } else {
                if (Gender < 1.000000000) {
                    return -0.135875046;
                } else {
                    return 1.222284670;
                }
            }
        }
    } else {
        if (BMI < 27.776929900) {
            if (BMI < 26.234567600) {
                if (Gender < 1.000000000) {
                    return 0.299662858;
                } else {
                    return 0.495485723;
                }
            } else {
                if (BMI < 26.312810900) {
                    return -0.073443584;
                } else {
                    return 0.121925034;
                }
            }
        } else {
            return 0.510310471;
        }
    }
}

inline double predict_tree_43(double Gender, double Age, double BMI) {
    if (Age < 49.000000000) {
        if (Age < 33.000000000) {
            if (Gender < 1.000000000) {
                if (Age < 31.000000000) {
                    return -0.206974849;
                } else {
                    return -0.012576412;
                }
            } else {
                if (Age < 31.000000000) {
                    return -0.059918012;
                } else {
                    return -0.207214803;
                }
            }
        } else {
            if (Age < 48.000000000) {
                if (Age < 47.000000000) {
                    return -0.028074423;
                } else {
                    return 0.231866807;
                }
            } else {
                if (Gender < 1.000000000) {
                    return -0.365110606;
                } else {
                    return -0.165458754;
                }
            }
        }
    } else {
        if (Gender < 1.000000000) {
            if (Age < 61.000000000) {
                if (Age < 50.000000000) {
                    return 0.516085982;
                } else {
                    return 0.072138935;
                }
            } else {
                if (Age < 66.000000000) {
                    return 0.480346173;
                } else {
                    return 0.192777470;
                }
            }
        } else {
            if (Age < 57.000000000) {
                if (Age < 56.000000000) {
                    return -0.029534167;
                } else {
                    return -0.434041172;
                }
            } else {
                if (Age < 60.000000000) {
                    return 0.298448950;
                } else {
                    return 0.006754546;
                }
            }
        }
    }
}

inline double predict_tree_44(double Gender, double Age, double BMI) {
    if (BMI < 26.528511000) {
        if (BMI < 26.423570600) {
            if (BMI < 21.877550100) {
                if (BMI < 21.545091600) {
                    return 0.066957772;
                } else {
                    return 0.241208717;
                }
            } else {
                if (BMI < 22.100290300) {
                    return -0.326857984;
                } else {
                    return -0.007162561;
                }
            }
        } else {
            if (Gender < 1.000000000) {
                return -0.627548873;
            } else {
                if (BMI < 26.485540400) {
                    return 0.369944751;
                } else {
                    return -0.414945364;
                }
            }
        }
    } else {
        if (Gender < 1.000000000) {
            if (BMI < 27.776929900) {
                if (BMI < 26.979530300) {
                    return 0.156435415;
                } else {
                    return 0.020589439;
                }
            } else {
                return 0.456688136;
            }
        } else {
            return 0.987421453;
        }
    }
}

inline double predict_tree_45(double Gender, double Age, double BMI) {
    if (BMI < 26.078971900) {
        if (BMI < 26.040952700) {
            if (BMI < 21.218317000) {
                if (BMI < 20.796730000) {
                    return -0.031855475;
                } else {
                    return 0.266310483;
                }
            } else {
                if (BMI < 22.282478300) {
                    return -0.141590238;
                } else {
                    return -0.017094484;
                }
            }
        } else {
            if (Gender < 1.000000000) {
                return -0.505056500;
            } else {
                return 1.016048910;
            }
        }
    } else {
        if (BMI < 26.282564200) {
            if (BMI < 26.196188000) {
                if (Gender < 1.000000000) {
                    return 0.081315517;
                } else {
                    return 0.227941543;
                }
            } else {
                if (Gender < 1.000000000) {
                    return 0.454342574;
                } else {
                    return -0.292173415;
                }
            }
        } else {
            if (BMI < 26.312810900) {
                if (BMI < 26.303619400) {
                    return -0.122392587;
                } else {
                    return -0.400706083;
                }
            } else {
                if (BMI < 26.595745100) {
                    return 0.215327621;
                } else {
                    return 0.038958937;
                }
            }
        }
    }
}

inline double predict_tree_46(double Gender, double Age, double BMI) {
    if (BMI < 21.218317000) {
        if (BMI < 20.796730000) {
            if (BMI < 20.504934300) {
                return 0.223055243;
            } else {
                return -0.225663826;
            }
        } else {
            if (BMI < 20.957170500) {
                return 0.199198484;
            } else {
                return 0.356712818;
            }
        }
    } else {
        if (BMI < 25.592210800) {
            if (BMI < 25.306932400) {
                if (BMI < 25.207756000) {
                    return -0.035243385;
                } else {
                    return 0.199840993;
                }
            } else {
                if (Gender < 1.000000000) {
                    return -0.163022041;
                } else {
                    return -0.546398818;
                }
            }
        } else {
            if (BMI < 26.014568300) {
                if (BMI < 25.951557200) {
                    return 0.042568944;
                } else {
                    return 0.397305816;
                }
            } else {
                if (BMI < 26.023048400) {
                    return -0.972928762;
                } else {
                    return 0.010657860;
                }
            }
        }
    }
}

inline double predict_tree_47(double Gender, double Age, double BMI) {
    if (BMI < 25.767219500) {
        if (BMI < 21.877550100) {
            if (BMI < 20.796730000) {
                if (BMI < 20.504934300) {
                    return 0.039384846;
                } else {
                    return -0.255312651;
                }
            } else {
                if (BMI < 21.218317000) {
                    return 0.291894883;
                } else {
                    return 0.098604187;
                }
            }
        } else {
            if (BMI < 22.230987500) {
                if (BMI < 22.189348200) {
                    return -0.171822205;
                } else {
                    return -0.610561550;
                }
            } else {
                if (BMI < 22.347782100) {
                    return 0.297146142;
                } else {
                    return -0.027393758;
                }
            }
        }
    } else {
        if (BMI < 25.772504800) {
            return 0.691986442;
        } else {
            if (BMI < 26.528511000) {
                if (BMI < 26.423570600) {
                    return 0.117577828;
                } else {
                    return -0.490209758;
                }
            } else {
                if (Gender < 1.000000000) {
                    return 0.141437635;
                } else {
                    return 1.089196090;
                }
            }
        }
    }
}

inline double predict_tree_48(double Gender, double Age, double BMI) {
    if (Age < 49.000000000) {
        if (Age < 37.000000000) {
            if (BMI < 26.851852400) {
                if (BMI < 26.312810900) {
                    return -0.082845047;
                } else {
                    return 0.244587794;
                }
            } else {
                if (BMI < 26.880950900) {
                    return -1.300490500;
                } else {
                    return -0.130638227;
                }
            }
        } else {
            if (Age < 48.000000000) {
                if (BMI < 25.181078000) {
                    return -0.014153461;
                } else {
                    return 0.143943131;
                }
            } else {
                if (BMI < 26.869806300) {
                    return -0.387601316;
                } else {
                    return 0.762710035;
                }
            }
        }
    } else {
        if (BMI < 21.773841900) {
            if (BMI < 21.718065300) {
                if (BMI < 21.612812000) {
                    return 0.545935690;
                } else {
                    return -0.709197760;
                }
            } else {
                if (Age < 51.000000000) {
                    return -0.014440047;
                } else {
                    return 2.463661910;
                }
            }
        } else {
            if (BMI < 21.989893000) {
                if (Age < 69.000000000) {
                    return -0.966825783;
                } else {
                    return 0.288791358;
                }
            } else {
                if (Age < 63.000000000) {
                    return 0.078648917;
                } else {
                    return 0.200413287;
                }
            }
        }
    }
}

inline double predict_tree_49(double Gender, double Age, double BMI) {
    if (Age < 44.000000000) {
        if (BMI < 21.218317000) {
            if (Age < 34.000000000) {
                if (BMI < 21.096191400) {
                    return -0.044282641;
                } else {
                    return 0.833836198;
                }
            } else {
                if (BMI < 21.096191400) {
                    return 1.163598180;
                } else {
                    return 0.128430322;
                }
            }
        } else {
            if (Age < 33.000000000) {
                if (BMI < 26.155626300) {
                    return -0.146108240;
                } else {
                    return 0.091930829;
                }
            } else {
                if (BMI < 26.555965400) {
                    return -0.031235980;
                } else {
                    return -0.341641754;
                }
            }
        }
    } else {
        if (Age < 73.000000000) {
            if (BMI < 23.147254900) {
                if (BMI < 22.862533600) {
                    return 0.064312823;
                } else {
                    return 0.628228068;
                }
            } else {
                if (BMI < 23.225431400) {
                    return -0.968144178;
                } else {
                    return 0.070920222;
                }
            }
        } else {
            if (BMI < 27.160493900) {
                if (BMI < 25.503616300) {
                    return 0.306602120;
                } else {
                    return -0.039141308;
                }
            } else {
                if (BMI < 27.450605400) {
                    return 1.540756230;
                } else {
                    return 0.355138004;
                }
            }
        }
    }
}

inline double predict_tree_50(double Gender, double Age, double BMI) {
    if (BMI < 20.796730000) {
        if (BMI < 20.504934300) {
            return 0.075164296;
        } else {
            return -0.726584017;
        }
    } else {
        if (BMI < 26.312810900) {
            if (BMI < 26.282564200) {
                if (BMI < 26.196188000) {
                    return 0.001884169;
                } else {
                    return 0.255057842;
                }
            } else {
                if (BMI < 26.296566000) {
                    return -0.649197757;
                } else {
                    return -0.159985408;
                }
            }
        } else {
            if (BMI < 26.315052000) {
                if (Gender < 1.000000000) {
                    return 1.065886380;
                } else {
                    return -0.226488531;
                }
            } else {
                if (BMI < 27.450605400) {
                    return 0.091910467;
                } else {
                    return -0.105426431;
                }
            }
        }
    }
}

inline double predict_tree_51(double Gender, double Age, double BMI) {
    if (Age < 49.000000000) {
        if (Age < 34.000000000) {
            if (Gender < 1.000000000) {
                if (Age < 21.000000000) {
                    return -0.346354187;
                } else {
                    return -0.144850448;
                }
            } else {
                if (Age < 25.000000000) {
                    return -0.045216594;
                } else {
                    return -0.101618059;
                }
            }
        } else {
            if (Age < 48.000000000) {
                if (Age < 47.000000000) {
                    return -0.012487025;
                } else {
                    return 0.284148455;
                }
            } else {
                if (Gender < 1.000000000) {
                    return -0.560943365;
                } else {
                    return -0.114535101;
                }
            }
        }
    } else {
        if (Gender < 1.000000000) {
            if (Age < 64.000000000) {
                if (Age < 50.000000000) {
                    return 0.520434678;
                } else {
                    return 0.094903074;
                }
            } else {
                if (Age < 66.000000000) {
                    return 0.697352827;
                } else {
                    return 0.262573719;
                }
            }
        } else {
            if (Age < 73.000000000) {
                if (Age < 70.000000000) {
                    return 0.056840513;
                } else {
                    return -0.292752564;
                }
            } else {
                if (Age < 75.000000000) {
                    return 0.580232084;
                } else {
                    return 0.024173791;
                }
            }
        }
    }
}

inline double predict_tree_52(double Gender, double Age, double BMI) {
    if (Age < 49.000000000) {
        if (Age < 22.000000000) {
            if (BMI < 26.528511000) {
                if (BMI < 26.023048400) {
                    return -0.201122835;
                } else {
                    return 0.612690926;
                }
            } else {
                return -0.935445249;
            }
        } else {
            if (BMI < 21.877550100) {
                if (Age < 31.000000000) {
                    return 0.278567493;
                } else {
                    return 0.001843031;
                }
            } else {
                if (BMI < 22.100290300) {
                    return -0.427607745;
                } else {
                    return -0.043398753;
                }
            }
        }
    } else {
        if (BMI < 21.773841900) {
            if (BMI < 21.718065300) {
                if (Age < 71.000000000) {
                    return 0.167958468;
                } else {
                    return 1.461181040;
                }
            } else {
                if (Age < 51.000000000) {
                    return -0.016922258;
                } else {
                    return 2.394683120;
                }
            }
        } else {
            if (BMI < 21.913805000) {
                if (Age < 53.000000000) {
                    return -1.487988110;
                } else {
                    return -0.490905076;
                }
            } else {
                if (BMI < 21.952478400) {
                    return 1.375120160;
                } else {
                    return 0.103338450;
                }
            }
        }
    }
}

inline double predict_tree_53(double Gender, double Age, double BMI) {
    if (Age < 49.000000000) {
        if (Age < 32.000000000) {
            if (Gender < 1.000000000) {
                if (Age < 21.000000000) {
                    return -0.268518418;
                } else {
                    return -0.128812373;
                }
            } else {
                if (Age < 23.000000000) {
                    return 0.041055679;
                } else {
                    return -0.080884576;
                }
            }
        } else {
            if (Age < 48.000000000) {
                if (Age < 47.000000000) {
                    return -0.030403188;
                } else {
                    return 0.259985954;
                }
            } else {
                if (Gender < 1.000000000) {
                    return -0.380354315;
                } else {
                    return -0.086356230;
                }
            }
        }
    } else {
        if (Gender < 1.000000000) {
            if (Age < 64.000000000) {
                if (Age < 50.000000000) {
                    return 0.530535698;
                } else {
                    return 0.106057540;
                }
            } else {
                if (Age < 66.000000000) {
                    return 0.506931543;
                } else {
                    return 0.271247476;
                }
            }
        } else {
            if (Age < 73.000000000) {
                if (Age < 70.000000000) {
                    return 0.067185566;
                } else {
                    return -0.243052423;
                }
            } else {
                if (Age < 75.000000000) {
                    return 0.515113294;
                } else {
                    return -0.000284097;
                }
            }
        }
    }
}

inline double predict_tree_54(double Gender, double Age, double BMI) {
    if (BMI < 25.767219500) {
        if (BMI < 25.762678100) {
            if (BMI < 25.306932400) {
                if (BMI < 24.989587800) {
                    return -0.013543121;
                } else {
                    return 0.211173490;
                }
            } else {
                if (BMI < 25.333334000) {
                    return -0.627874613;
                } else {
                    return -0.086719565;
                }
            }
        } else {
            if (Gender < 1.000000000) {
                return -0.817444921;
            } else {
                return 0.671278894;
            }
        }
    } else {
        if (BMI < 25.795917500) {
            if (Gender < 1.000000000) {
                if (BMI < 25.772504800) {
                    return 0.555541456;
                } else {
                    return 0.042574279;
                }
            } else {
                return 1.124711990;
            }
        } else {
            if (BMI < 25.854639100) {
                if (Gender < 1.000000000) {
                    return -0.208058372;
                } else {
                    return -0.552734852;
                }
            } else {
                if (BMI < 25.992439300) {
                    return 0.269167781;
                } else {
                    return 0.053085163;
                }
            }
        }
    }
}

inline double predict_tree_55(double Gender, double Age, double BMI) {
    if (Age < 32.000000000) {
        if (Gender < 1.000000000) {
            if (Age < 28.000000000) {
                if (Age < 26.000000000) {
                    return -0.208516911;
                } else {
                    return -0.292567909;
                }
            } else {
                if (Age < 29.000000000) {
                    return 0.075239278;
                } else {
                    return -0.162977755;
                }
            }
        } else {
            if (Age < 26.000000000) {
                if (Age < 25.000000000) {
                    return -0.087991066;
                } else {
                    return -0.279909015;
                }
            } else {
                if (Age < 31.000000000) {
                    return 0.022745859;
                } else {
                    return -0.153813705;
                }
            }
        }
    } else {
        if (Age < 57.000000000) {
            if (Age < 55.000000000) {
                if (Age < 54.000000000) {
                    return 0.010705017;
                } else {
                    return 0.313789368;
                }
            } else {
                if (Gender < 1.000000000) {
                    return -0.277969331;
                } else {
                    return -0.091473065;
                }
            }
        } else {
            if (Gender < 1.000000000) {
                if (Age < 77.000000000) {
                    return 0.175067231;
                } else {
                    return 0.465824395;
                }
            } else {
                if (Age < 76.000000000) {
                    return 0.135019302;
                } else {
                    return -0.151195690;
                }
            }
        }
    }
}

inline double predict_tree_56(double Gender, double Age, double BMI) {
    if (Age < 57.000000000) {
        if (Age < 22.000000000) {
            if (Gender < 1.000000000) {
                if (Age < 21.000000000) {
                    return -0.388556242;
                } else {
                    return -0.162056074;
                }
            } else {
                if (Age < 21.000000000) {
                    return -0.142973244;
                } else {
                    return -0.106308751;
                }
            }
        } else {
            if (Age < 41.000000000) {
                if (Gender < 1.000000000) {
                    return -0.088649146;
                } else {
                    return -0.014586208;
                }
            } else {
                if (Age < 42.000000000) {
                    return 0.312894821;
                } else {
                    return -0.018410772;
                }
            }
        }
    } else {
        if (Gender < 1.000000000) {
            if (Age < 64.000000000) {
                if (Age < 63.000000000) {
                    return 0.153559282;
                } else {
                    return -0.205973253;
                }
            } else {
                if (Age < 66.000000000) {
                    return 0.597582519;
                } else {
                    return 0.227660418;
                }
            }
        } else {
            if (Age < 59.000000000) {
                if (Age < 58.000000000) {
                    return 0.213441402;
                } else {
                    return 0.349830776;
                }
            } else {
                if (Age < 63.000000000) {
                    return -0.148818314;
                } else {
                    return 0.055700641;
                }
            }
        }
    }
}

inline double predict_tree_57(double Gender, double Age, double BMI) {
    if (Age < 58.000000000) {
        if (Age < 34.000000000) {
            if (BMI < 22.862533600) {
                if (BMI < 22.714681600) {
                    return -0.064996719;
                } else {
                    return 0.277521104;
                }
            } else {
                if (BMI < 22.991689700) {
                    return -0.652977943;
                } else {
                    return -0.122922972;
                }
            }
        } else {
            if (BMI < 26.423570600) {
                if (BMI < 26.315052000) {
                    return 0.014031344;
                } else {
                    return 0.615973353;
                }
            } else {
                if (Age < 39.000000000) {
                    return -0.831594288;
                } else {
                    return -0.012510149;
                }
            }
        }
    } else {
        if (BMI < 23.533042900) {
            if (Age < 59.000000000) {
                if (BMI < 21.718065300) {
                    return -0.705094218;
                } else {
                    return 1.451386570;
                }
            } else {
                if (BMI < 23.335466400) {
                    return 0.074823722;
                } else {
                    return 0.586007118;
                }
            }
        } else {
            if (BMI < 23.833004000) {
                if (BMI < 23.808797800) {
                    return -0.304564804;
                } else {
                    return -1.672304030;
                }
            } else {
                if (BMI < 23.922422400) {
                    return 1.034977200;
                } else {
                    return 0.126581490;
                }
            }
        }
    }
}

inline double predict_tree_58(double Gender, double Age, double BMI) {
    if (BMI < 25.767219500) {
        if (BMI < 25.762678100) {
            if (BMI < 21.952478400) {
                if (BMI < 21.913805000) {
                    return 0.063283004;
                } else {
                    return 0.641836941;
                }
            } else {
                if (BMI < 22.100290300) {
                    return -0.446891427;
                } else {
                    return 0.002439437;
                }
            }
        } else {
            if (Gender < 1.000000000) {
                return -0.693920553;
            } else {
                return 0.652260065;
            }
        }
    } else {
        if (BMI < 25.795917500) {
            if (Gender < 1.000000000) {
                if (BMI < 25.772504800) {
                    return 0.689935267;
                } else {
                    return 0.096758611;
                }
            } else {
                return 1.090742470;
            }
        } else {
            if (BMI < 25.854639100) {
                if (Gender < 1.000000000) {
                    return -0.318391651;
                } else {
                    return -0.910725057;
                }
            } else {
                if (BMI < 25.992439300) {
                    return 0.268337697;
                } else {
                    return 0.052457530;
                }
            }
        }
    }
}

inline double predict_tree_59(double Gender, double Age, double BMI) {
    if (Age < 57.000000000) {
        if (Age < 29.000000000) {
            if (BMI < 26.312810900) {
                if (BMI < 21.913805000) {
                    return 0.083295114;
                } else {
                    return -0.163852215;
                }
            } else {
                if (BMI < 26.332889600) {
                    return 0.908781707;
                } else {
                    return 0.187586844;
                }
            }
        } else {
            if (BMI < 26.869806300) {
                if (BMI < 26.851852400) {
                    return -0.010183820;
                } else {
                    return 0.861872256;
                }
            } else {
                if (Age < 38.000000000) {
                    return -0.888281703;
                } else {
                    return -0.064647861;
                }
            }
        }
    } else {
        if (BMI < 24.670375800) {
            if (BMI < 24.302486400) {
                if (BMI < 24.251277900) {
                    return 0.164849564;
                } else {
                    return 1.259768840;
                }
            } else {
                if (Age < 72.000000000) {
                    return -0.580039024;
                } else {
                    return 0.453761965;
                }
            }
        } else {
            if (Age < 70.000000000) {
                if (BMI < 24.835645700) {
                    return 1.094347830;
                } else {
                    return 0.255039454;
                }
            } else {
                if (BMI < 26.592670400) {
                    return -0.088563561;
                } else {
                    return 0.344229460;
                }
            }
        }
    }
}

inline double predict_tree_60(double Gender, double Age, double BMI) {
    if (Age < 63.000000000) {
        if (Age < 41.000000000) {
            if (Gender < 1.000000000) {
                if (Age < 21.000000000) {
                    return -0.301744372;
                } else {
                    return -0.091609813;
                }
            } else {
                if (Age < 22.000000000) {
                    return -0.106126457;
                } else {
                    return -0.011012700;
                }
            }
        } else {
            if (Age < 42.000000000) {
                if (Gender < 1.000000000) {
                    return 0.031492140;
                } else {
                    return 0.339975268;
                }
            } else {
                if (Gender < 1.000000000) {
                    return 0.057970393;
                } else {
                    return -0.045580067;
                }
            }
        }
    } else {
        if (Age < 65.000000000) {
            if (Age < 64.000000000) {
                if (Gender < 1.000000000) {
                    return -0.195847780;
                } else {
                    return 0.338443667;
                }
            } else {
                if (Gender < 1.000000000) {
                    return 0.777347445;
                } else {
                    return 0.147324219;
                }
            }
        } else {
            if (Age < 72.000000000) {
                if (Age < 70.000000000) {
                    return 0.104653336;
                } else {
                    return -0.024787132;
                }
            } else {
                if (Age < 74.000000000) {
                    return 0.311479032;
                } else {
                    return 0.128904626;
                }
            }
        }
    }
}

inline double predict_tree_61(double Gender, double Age, double BMI) {
    if (Age < 57.000000000) {
        if (Age < 56.000000000) {
            if (Age < 49.000000000) {
                if (Gender < 1.000000000) {
                    return -0.087403580;
                } else {
                    return -0.017349960;
                }
            } else {
                if (Age < 50.000000000) {
                    return 0.306449831;
                } else {
                    return 0.016003212;
                }
            }
        } else {
            return -0.337391227;
        }
    } else {
        if (Age < 78.000000000) {
            if (Age < 64.000000000) {
                if (Age < 62.000000000) {
                    return 0.123997711;
                } else {
                    return -0.089534193;
                }
            } else {
                if (Age < 65.000000000) {
                    return 0.400084436;
                } else {
                    return 0.133956105;
                }
            }
        } else {
            if (Age < 79.000000000) {
                if (Gender < 1.000000000) {
                    return -0.303623676;
                } else {
                    return -0.058007848;
                }
            } else {
                if (Gender < 1.000000000) {
                    return -0.024011828;
                } else {
                    return 0.016040444;
                }
            }
        }
    }
}

inline double predict_tree_62(double Gender, double Age, double BMI) {
    if (Age < 54.000000000) {
        if (Age < 52.000000000) {
            if (Age < 49.000000000) {
                if (Age < 48.000000000) {
                    return -0.036664963;
                } else {
                    return -0.256529868;
                }
            } else {
                if (Age < 50.000000000) {
                    return 0.256614894;
                } else {
                    return 0.085364722;
                }
            }
        } else {
            if (Age < 53.000000000) {
                if (Gender < 1.000000000) {
                    return -0.137061387;
                } else {
                    return -0.491950244;
                }
            } else {
                if (Gender < 1.000000000) {
                    return -0.334767550;
                } else {
                    return -0.046187986;
                }
            }
        }
    } else {
        if (Gender < 1.000000000) {
            if (Age < 55.000000000) {
                return 0.715281308;
            } else {
                if (Age < 56.000000000) {
                    return -0.341966212;
                } else {
                    return 0.142692715;
                }
            }
        } else {
            if (Age < 76.000000000) {
                if (Age < 73.000000000) {
                    return -0.001531065;
                } else {
                    return 0.338483959;
                }
            } else {
                if (Age < 78.000000000) {
                    return -0.262458444;
                } else {
                    return -0.034438308;
                }
            }
        }
    }
}

inline double predict_tree_63(double Gender, double Age, double BMI) {
    if (Age < 57.000000000) {
        if (Age < 34.000000000) {
            if (Gender < 1.000000000) {
                if (Age < 22.000000000) {
                    return -0.222346202;
                } else {
                    return -0.078796364;
                }
            } else {
                if (Age < 31.000000000) {
                    return -0.017650310;
                } else {
                    return -0.165840343;
                }
            }
        } else {
            if (Gender < 1.000000000) {
                if (Age < 41.000000000) {
                    return -0.048373856;
                } else {
                    return 0.051487580;
                }
            } else {
                if (Age < 56.000000000) {
                    return -0.039427690;
                } else {
                    return -0.437314391;
                }
            }
        }
    } else {
        if (Gender < 1.000000000) {
            if (Age < 61.000000000) {
                if (Age < 60.000000000) {
                    return 0.020004913;
                } else {
                    return -0.201718405;
                }
            } else {
                if (Age < 74.000000000) {
                    return 0.264257550;
                } else {
                    return -0.005598812;
                }
            }
        } else {
            if (Age < 60.000000000) {
                if (Age < 58.000000000) {
                    return 0.055529851;
                } else {
                    return 0.231241450;
                }
            } else {
                if (Age < 63.000000000) {
                    return -0.170350417;
                } else {
                    return 0.024510665;
                }
            }
        }
    }
}

inline double predict_tree_64(double Gender, double Age, double BMI) {
    if (Age < 57.000000000) {
        if (Age < 29.000000000) {
            if (Age < 22.000000000) {
                if (Gender < 1.000000000) {
                    return -0.192504808;
                } else {
                    return -0.070517510;
                }
            } else {
                if (Age < 23.000000000) {
                    return 0.025126042;
                } else {
                    return -0.089938127;
                }
            }
        } else {
            if (Age < 55.000000000) {
                if (Gender < 1.000000000) {
                    return 0.050616778;
                } else {
                    return -0.033303462;
                }
            } else {
                if (Gender < 1.000000000) {
                    return -0.284397691;
                } else {
                    return -0.074754760;
                }
            }
        }
    } else {
        if (Gender < 1.000000000) {
            if (Age < 75.000000000) {
                if (Age < 74.000000000) {
                    return 0.156002253;
                } else {
                    return -0.320817024;
                }
            } else {
                if (Age < 79.000000000) {
                    return 0.411020190;
                } else {
                    return 0.274423540;
                }
            }
        } else {
            if (Age < 60.000000000) {
                if (Age < 58.000000000) {
                    return 0.109984122;
                } else {
                    return 0.163710609;
                }
            } else {
                if (Age < 61.000000000) {
                    return -0.559193552;
                } else {
                    return 0.002534605;
                }
            }
        }
    }
}

inline double predict_tree_65(double Gender, double Age, double BMI) {
    if (Age < 41.000000000) {
        if (Age < 21.000000000) {
            if (Gender < 1.000000000) {
                return -0.221506879;
            } else {
                return -0.182217285;
            }
        } else {
            if (Gender < 1.000000000) {
                if (Age < 32.000000000) {
                    return -0.110639840;
                } else {
                    return -0.038253095;
                }
            } else {
                if (Age < 37.000000000) {
                    return -0.031011624;
                } else {
                    return 0.048293769;
                }
            }
        }
    } else {
        if (Gender < 1.000000000) {
            if (Age < 61.000000000) {
                if (Age < 60.000000000) {
                    return 0.076785564;
                } else {
                    return -0.452412754;
                }
            } else {
                if (Age < 62.000000000) {
                    return 0.560790360;
                } else {
                    return 0.166385755;
                }
            }
        } else {
            if (Age < 42.000000000) {
                return 0.383931011;
            } else {
                if (Age < 44.000000000) {
                    return -0.223978341;
                } else {
                    return 0.013752829;
                }
            }
        }
    }
}

inline double predict_tree_66(double Gender, double Age, double BMI) {
    if (Age < 57.000000000) {
        if (Age < 28.000000000) {
            if (Gender < 1.000000000) {
                if (Age < 26.000000000) {
                    return -0.071721487;
                } else {
                    return -0.231999710;
                }
            } else {
                if (Age < 23.000000000) {
                    return 0.018418970;
                } else {
                    return -0.056913435;
                }
            }
        } else {
            if (Gender < 1.000000000) {
                if (Age < 47.000000000) {
                    return -0.003336129;
                } else {
                    return 0.075830087;
                }
            } else {
                if (Age < 30.000000000) {
                    return 0.123491541;
                } else {
                    return -0.039924316;
                }
            }
        }
    } else {
        if (Gender < 1.000000000) {
            if (Age < 77.000000000) {
                if (Age < 66.000000000) {
                    return 0.219790295;
                } else {
                    return 0.050702576;
                }
            } else {
                if (Age < 78.000000000) {
                    return 0.474475086;
                } else {
                    return 0.310921729;
                }
            }
        } else {
            if (Age < 60.000000000) {
                if (Age < 58.000000000) {
                    return 0.136994168;
                } else {
                    return 0.241569266;
                }
            } else {
                if (Age < 61.000000000) {
                    return -0.497560918;
                } else {
                    return -0.001571216;
                }
            }
        }
    }
}

inline double predict_tree_67(double Gender, double Age, double BMI) {
    if (BMI < 21.367521300) {
        if (BMI < 21.303949400) {
            if (BMI < 21.218317000) {
                if (BMI < 20.957170500) {
                    return 0.069682531;
                } else {
                    return 0.317198336;
                }
            } else {
                return -0.227725118;
            }
        } else {
            return 0.392081141;
        }
    } else {
        if (BMI < 22.230987500) {
            if (BMI < 22.212966900) {
                if (BMI < 21.952478400) {
                    return 0.000089597;
                } else {
                    return -0.174043059;
                }
            } else {
                return -0.709559023;
            }
        } else {
            if (BMI < 22.347782100) {
                if (Gender < 1.000000000) {
                    return -0.764624476;
                } else {
                    return 0.446441263;
                }
            } else {
                if (BMI < 23.147254900) {
                    return 0.059265275;
                } else {
                    return -0.014655702;
                }
            }
        }
    }
}

inline double predict_tree_68(double Gender, double Age, double BMI) {
    if (Age < 41.000000000) {
        if (BMI < 26.555965400) {
            if (BMI < 26.312810900) {
                if (BMI < 26.303619400) {
                    return -0.051065449;
                } else {
                    return -0.783388674;
                }
            } else {
                if (Age < 38.000000000) {
                    return 0.505582869;
                } else {
                    return -0.299339473;
                }
            }
        } else {
            if (Age < 35.000000000) {
                if (BMI < 26.827421200) {
                    return 0.206085965;
                } else {
                    return -0.264109761;
                }
            } else {
                if (BMI < 26.592670400) {
                    return -1.583405730;
                } else {
                    return -0.582224965;
                }
            }
        }
    } else {
        if (Age < 72.000000000) {
            if (Age < 71.000000000) {
                if (BMI < 22.230987500) {
                    return -0.158894092;
                } else {
                    return 0.068483189;
                }
            } else {
                if (BMI < 26.880950900) {
                    return -0.088175923;
                } else {
                    return -1.421682600;
                }
            }
        } else {
            if (BMI < 21.829952200) {
                return 1.792266490;
            } else {
                if (BMI < 26.592670400) {
                    return 0.078369856;
                } else {
                    return 0.530719697;
                }
            }
        }
    }
}

inline double predict_tree_69(double Gender, double Age, double BMI) {
    if (Age < 47.000000000) {
        if (Age < 32.000000000) {
            if (Age < 31.000000000) {
                if (Gender < 1.000000000) {
                    return -0.083233982;
                } else {
                    return -0.033095732;
                }
            } else {
                return -0.152340651;
            }
        } else {
            if (Age < 46.000000000) {
                if (Age < 45.000000000) {
                    return -0.024309078;
                } else {
                    return 0.174501702;
                }
            } else {
                if (Gender < 1.000000000) {
                    return -0.218436167;
                } else {
                    return -0.084672414;
                }
            }
        }
    } else {
        if (Gender < 1.000000000) {
            if (Age < 48.000000000) {
                return 0.486730248;
            } else {
                if (Age < 49.000000000) {
                    return -0.407662719;
                } else {
                    return 0.130952150;
                }
            }
        } else {
            if (Age < 75.000000000) {
                if (Age < 74.000000000) {
                    return 0.048496444;
                } else {
                    return 0.538531482;
                }
            } else {
                if (Age < 79.000000000) {
                    return -0.104677744;
                } else {
                    return -0.259255946;
                }
            }
        }
    }
}

inline double predict_tree_70(double Gender, double Age, double BMI) {
    if (Age < 64.000000000) {
        if (Age < 29.000000000) {
            if (Age < 21.000000000) {
                if (Gender < 1.000000000) {
                    return -0.201465055;
                } else {
                    return -0.090802506;
                }
            } else {
                if (Age < 24.000000000) {
                    return 0.003183075;
                } else {
                    return -0.088845000;
                }
            }
        } else {
            if (Gender < 1.000000000) {
                if (Age < 31.000000000) {
                    return -0.170153201;
                } else {
                    return 0.005087305;
                }
            } else {
                if (Age < 31.000000000) {
                    return 0.199877158;
                } else {
                    return 0.007626368;
                }
            }
        }
    } else {
        if (Gender < 1.000000000) {
            if (Age < 66.000000000) {
                if (Age < 65.000000000) {
                    return 0.597577512;
                } else {
                    return 0.387003273;
                }
            } else {
                if (Age < 67.000000000) {
                    return -0.515569091;
                } else {
                    return 0.199724898;
                }
            }
        } else {
            if (Age < 76.000000000) {
                if (Age < 73.000000000) {
                    return -0.028156949;
                } else {
                    return 0.402794898;
                }
            } else {
                if (Age < 78.000000000) {
                    return -0.056005642;
                } else {
                    return -0.177051678;
                }
            }
        }
    }
}

inline double predict_tree_71(double Gender, double Age, double BMI) {
    if (Age < 61.000000000) {
        if (BMI < 27.776929900) {
            if (BMI < 27.160493900) {
                if (BMI < 27.137414900) {
                    return -0.026782755;
                } else {
                    return 0.749931693;
                }
            } else {
                if (Age < 60.000000000) {
                    return -0.190288156;
                } else {
                    return -1.618850110;
                }
            }
        } else {
            if (Age < 53.000000000) {
                if (Age < 40.000000000) {
                    return -0.366916537;
                } else {
                    return 1.776737210;
                }
            } else {
                if (Age < 57.000000000) {
                    return -1.557306170;
                } else {
                    return 1.198028560;
                }
            }
        }
    } else {
        if (BMI < 23.046875000) {
            if (Age < 65.000000000) {
                if (BMI < 22.145328500) {
                    return 0.070840120;
                } else {
                    return -0.754155874;
                }
            } else {
                if (Age < 75.000000000) {
                    return 0.344148129;
                } else {
                    return -0.633242905;
                }
            }
        } else {
            if (BMI < 23.147254900) {
                if (Age < 78.000000000) {
                    return 1.159096000;
                } else {
                    return -0.170524195;
                }
            } else {
                if (BMI < 23.225431400) {
                    return -0.967949331;
                } else {
                    return 0.140503258;
                }
            }
        }
    }
}

inline double predict_tree_72(double Gender, double Age, double BMI) {
    if (Age < 32.000000000) {
        if (Gender < 1.000000000) {
            if (Age < 25.000000000) {
                if (Age < 24.000000000) {
                    return -0.156924859;
                } else {
                    return -0.345333308;
                }
            } else {
                if (Age < 29.000000000) {
                    return -0.007723004;
                } else {
                    return -0.089878574;
                }
            }
        } else {
            if (Age < 23.000000000) {
                if (Age < 22.000000000) {
                    return -0.030019168;
                } else {
                    return 0.194536582;
                }
            } else {
                if (Age < 26.000000000) {
                    return -0.151246414;
                } else {
                    return -0.025904888;
                }
            }
        }
    } else {
        if (Age < 73.000000000) {
            if (Age < 71.000000000) {
                if (Age < 61.000000000) {
                    return 0.006435041;
                } else {
                    return 0.090308048;
                }
            } else {
                if (Gender < 1.000000000) {
                    return -0.015213471;
                } else {
                    return -0.283584863;
                }
            }
        } else {
            if (Gender < 1.000000000) {
                if (Age < 77.000000000) {
                    return 0.130266324;
                } else {
                    return 0.405971885;
                }
            } else {
                if (Age < 76.000000000) {
                    return 0.261285126;
                } else {
                    return -0.229061291;
                }
            }
        }
    }
}

inline double predict_tree_73(double Gender, double Age, double BMI) {
    if (Age < 22.000000000) {
        if (BMI < 23.225431400) {
            if (BMI < 22.862533600) {
                if (BMI < 22.790329000) {
                    return -0.234570846;
                } else {
                    return 0.874617338;
                }
            } else {
                if (BMI < 23.030044600) {
                    return -1.318128350;
                } else {
                    return -0.224873140;
                }
            }
        } else {
            if (BMI < 23.733238200) {
                if (BMI < 23.711845400) {
                    return 0.151226521;
                } else {
                    return 1.248525620;
                }
            } else {
                if (BMI < 24.017253900) {
                    return -0.510505259;
                } else {
                    return -0.000254695;
                }
            }
        }
    } else {
        if (BMI < 21.218317000) {
            if (Age < 23.000000000) {
                if (BMI < 20.796730000) {
                    return -0.606763721;
                } else {
                    return 0.265991330;
                }
            } else {
                if (Age < 26.000000000) {
                    return 0.835356653;
                } else {
                    return 0.184332475;
                }
            }
        } else {
            if (BMI < 21.545091600) {
                if (Age < 47.000000000) {
                    return -0.371945560;
                } else {
                    return 0.469029456;
                }
            } else {
                if (BMI < 21.641273500) {
                    return 0.460836679;
                } else {
                    return 0.013346776;
                }
            }
        }
    }
}

inline double predict_tree_74(double Gender, double Age, double BMI) {
    if (Age < 57.000000000) {
        if (Age < 56.000000000) {
            if (Age < 43.000000000) {
                if (Age < 42.000000000) {
                    return -0.040898021;
                } else {
                    return -0.239049807;
                }
            } else {
                if (Age < 46.000000000) {
                    return 0.097048730;
                } else {
                    return -0.011750724;
                }
            }
        } else {
            if (Gender < 1.000000000) {
                return -0.184217513;
            } else {
                return -0.291829109;
            }
        }
    } else {
        if (Age < 79.000000000) {
            if (Age < 76.000000000) {
                if (Age < 72.000000000) {
                    return 0.071484223;
                } else {
                    return 0.151645467;
                }
            } else {
                if (Gender < 1.000000000) {
                    return -0.009410000;
                } else {
                    return -0.076267272;
                }
            }
        } else {
            if (Gender < 1.000000000) {
                return 0.330503076;
            } else {
                return 0.133729964;
            }
        }
    }
}

inline double predict_tree_75(double Gender, double Age, double BMI) {
    if (Age < 47.000000000) {
        if (Gender < 1.000000000) {
            if (Age < 21.000000000) {
                return -0.310193360;
            } else {
                if (Age < 32.000000000) {
                    return -0.088218331;
                } else {
                    return -0.028428327;
                }
            }
        } else {
            if (Age < 44.000000000) {
                if (Age < 42.000000000) {
                    return 0.010088237;
                } else {
                    return -0.326927751;
                }
            } else {
                if (Age < 46.000000000) {
                    return 0.144172668;
                } else {
                    return -0.014947103;
                }
            }
        }
    } else {
        if (Age < 48.000000000) {
            if (Gender < 1.000000000) {
                return 0.725624084;
            } else {
                return -0.032015640;
            }
        } else {
            if (Age < 49.000000000) {
                if (Gender < 1.000000000) {
                    return -0.426133364;
                } else {
                    return 0.017114645;
                }
            } else {
                if (Age < 76.000000000) {
                    return 0.067495331;
                } else {
                    return -0.109149672;
                }
            }
        }
    }
}

inline double predict_tree_76(double Gender, double Age, double BMI) {
    if (Age < 32.000000000) {
        if (BMI < 26.196188000) {
            if (BMI < 25.401701000) {
                if (BMI < 24.618103000) {
                    return -0.111550227;
                } else {
                    return 0.068676338;
                }
            } else {
                if (BMI < 25.469387100) {
                    return -0.962367237;
                } else {
                    return -0.155984744;
                }
            }
        } else {
            if (Age < 25.000000000) {
                if (BMI < 26.234567600) {
                    return 0.785167217;
                } else {
                    return -0.280569524;
                }
            } else {
                if (Age < 29.000000000) {
                    return 0.770818532;
                } else {
                    return -0.177531064;
                }
            }
        }
    } else {
        if (BMI < 20.504934300) {
            return 1.063583610;
        } else {
            if (Age < 57.000000000) {
                if (BMI < 26.423570600) {
                    return 0.022451470;
                } else {
                    return -0.174838886;
                }
            } else {
                if (BMI < 23.555555300) {
                    return 0.183186144;
                } else {
                    return 0.062122624;
                }
            }
        }
    }
}

inline double predict_tree_77(double Gender, double Age, double BMI) {
    if (Age < 61.000000000) {
        if (BMI < 22.598140700) {
            if (Age < 53.000000000) {
                if (Age < 22.000000000) {
                    return -0.180951968;
                } else {
                    return 0.071335219;
                }
            } else {
                if (BMI < 22.375679000) {
                    return 0.251070768;
                } else {
                    return 1.072824240;
                }
            }
        } else {
            if (BMI < 27.160493900) {
                if (BMI < 26.880950900) {
                    return -0.031897765;
                } else {
                    return 0.406013131;
                }
            } else {
                if (BMI < 27.776929900) {
                    return -0.442041397;
                } else {
                    return 0.491359919;
                }
            }
        }
    } else {
        if (BMI < 27.450605400) {
            if (BMI < 27.173914000) {
                if (BMI < 21.096191400) {
                    return 1.612420200;
                } else {
                    return 0.065046392;
                }
            } else {
                if (Age < 65.000000000) {
                    return -0.008168579;
                } else {
                    return 1.384486680;
                }
            }
        } else {
            if (Age < 77.000000000) {
                if (Age < 64.000000000) {
                    return -1.146823530;
                } else {
                    return -0.449093729;
                }
            } else {
                if (Age < 78.000000000) {
                    return 1.292205450;
                } else {
                    return -0.038620409;
                }
            }
        }
    }
}

inline double predict_tree_78(double Gender, double Age, double BMI) {
    if (BMI < 23.597003900) {
        if (Gender < 1.000000000) {
            if (BMI < 22.790329000) {
                if (BMI < 22.720438000) {
                    return -0.662220120;
                } else {
                    return -1.784191130;
                }
            } else {
                if (BMI < 22.829963700) {
                    return 1.539207460;
                } else {
                    return -0.218476236;
                }
            }
        } else {
            if (BMI < 23.555555300) {
                if (BMI < 20.504934300) {
                    return 0.270621598;
                } else {
                    return -0.019665998;
                }
            } else {
                return -0.402430713;
            }
        }
    } else {
        if (BMI < 23.634033200) {
            if (Gender < 1.000000000) {
                if (BMI < 23.620288800) {
                    return 1.682603000;
                } else {
                    return 0.289334685;
                }
            } else {
                return 0.551943541;
            }
        } else {
            if (BMI < 23.795360600) {
                if (BMI < 23.711845400) {
                    return -0.245614231;
                } else {
                    return 0.411799997;
                }
            } else {
                if (BMI < 23.849777200) {
                    return -0.385471791;
                } else {
                    return 0.002223295;
                }
            }
        }
    }
}

inline double predict_tree_79(double Gender, double Age, double BMI) {
    if (Age < 63.000000000) {
        if (Age < 26.000000000) {
            if (BMI < 24.242424000) {
                if (BMI < 24.017253900) {
                    return -0.066473961;
                } else {
                    return 0.415613204;
                }
            } else {
                if (BMI < 24.618103000) {
                    return -0.445019454;
                } else {
                    return -0.116230950;
                }
            }
        } else {
            if (BMI < 27.160493900) {
                if (BMI < 23.597003900) {
                    return -0.057743184;
                } else {
                    return 0.021830434;
                }
            } else {
                if (Age < 53.000000000) {
                    return 0.007045835;
                } else {
                    return -0.615828514;
                }
            }
        }
    } else {
        if (BMI < 25.306932400) {
            if (BMI < 22.862533600) {
                if (Age < 75.000000000) {
                    return 0.165496856;
                } else {
                    return -1.093606710;
                }
            } else {
                if (BMI < 23.147254900) {
                    return 0.887320995;
                } else {
                    return 0.203042522;
                }
            }
        } else {
            if (BMI < 25.381467800) {
                if (Age < 77.000000000) {
                    return -1.336970090;
                } else {
                    return 0.287614793;
                }
            } else {
                if (BMI < 27.173914000) {
                    return -0.035821930;
                } else {
                    return 0.397581905;
                }
            }
        }
    }
}

inline double predict_tree_80(double Gender, double Age, double BMI) {
    if (Age < 64.000000000) {
        if (BMI < 21.952478400) {
            if (Age < 22.000000000) {
                if (BMI < 21.877550100) {
                    return -0.300699651;
                } else {
                    return 0.897713423;
                }
            } else {
                if (Age < 28.000000000) {
                    return 0.371492386;
                } else {
                    return 0.115018986;
                }
            }
        } else {
            if (BMI < 22.100290300) {
                if (Age < 24.000000000) {
                    return 0.054120325;
                } else {
                    return -0.580238342;
                }
            } else {
                if (BMI < 22.582708400) {
                    return 0.167431727;
                } else {
                    return -0.026333610;
                }
            }
        }
    } else {
        if (Age < 65.000000000) {
            if (BMI < 23.939481700) {
                if (BMI < 23.711845400) {
                    return -0.058970556;
                } else {
                    return -1.311857220;
                }
            } else {
                if (BMI < 26.595745100) {
                    return 0.883297026;
                } else {
                    return 0.115827411;
                }
            }
        } else {
            if (BMI < 26.880950900) {
                if (BMI < 26.595745100) {
                    return 0.068259835;
                } else {
                    return 1.045711520;
                }
            } else {
                if (BMI < 27.173914000) {
                    return -0.749944448;
                } else {
                    return 0.283453941;
                }
            }
        }
    }
}

inline double predict_tree_81(double Gender, double Age, double BMI) {
    if (Age < 61.000000000) {
        if (Age < 60.000000000) {
            if (Age < 35.000000000) {
                if (Gender < 1.000000000) {
                    return -0.056344803;
                } else {
                    return -0.014144670;
                }
            } else {
                if (Age < 51.000000000) {
                    return 0.024554689;
                } else {
                    return -0.053271197;
                }
            }
        } else {
            if (Gender < 1.000000000) {
                return -0.203615531;
            } else {
                return -0.391401231;
            }
        }
    } else {
        if (Gender < 1.000000000) {
            if (Age < 62.000000000) {
                return 0.608103871;
            } else {
                if (Age < 64.000000000) {
                    return -0.069613405;
                } else {
                    return 0.160168320;
                }
            }
        } else {
            if (Age < 76.000000000) {
                if (Age < 73.000000000) {
                    return -0.019022232;
                } else {
                    return 0.263372809;
                }
            } else {
                if (Age < 78.000000000) {
                    return -0.156008407;
                } else {
                    return -0.091662213;
                }
            }
        }
    }
}

inline double predict_tree_82(double Gender, double Age, double BMI) {
    if (Age < 61.000000000) {
        if (Age < 60.000000000) {
            if (Age < 22.000000000) {
                if (Gender < 1.000000000) {
                    return -0.131151944;
                } else {
                    return -0.000017247;
                }
            } else {
                if (Age < 23.000000000) {
                    return 0.074914746;
                } else {
                    return -0.012213379;
                }
            }
        } else {
            if (Gender < 1.000000000) {
                return 0.112973876;
            } else {
                return -0.533857703;
            }
        }
    } else {
        if (Age < 65.000000000) {
            if (Age < 64.000000000) {
                if (Gender < 1.000000000) {
                    return 0.240668267;
                } else {
                    return 0.113062501;
                }
            } else {
                if (Gender < 1.000000000) {
                    return 0.405297995;
                } else {
                    return 0.276587188;
                }
            }
        } else {
            if (Age < 75.000000000) {
                if (Age < 73.000000000) {
                    return 0.048803866;
                } else {
                    return 0.176209196;
                }
            } else {
                if (Gender < 1.000000000) {
                    return 0.045365293;
                } else {
                    return -0.025028393;
                }
            }
        }
    }
}

inline double predict_tree_83(double Gender, double Age, double BMI) {
    if (BMI < 21.877550100) {
        if (BMI < 21.718065300) {
            if (BMI < 21.641273500) {
                if (BMI < 21.545091600) {
                    return -0.001929011;
                } else {
                    return 0.225744650;
                }
            } else {
                return -0.272779107;
            }
        } else {
            if (BMI < 21.829952200) {
                return 0.312784523;
            } else {
                return 0.091394015;
            }
        }
    } else {
        if (BMI < 22.100290300) {
            if (BMI < 21.952478400) {
                if (BMI < 21.913805000) {
                    return -0.365447342;
                } else {
                    return 0.369899422;
                }
            } else {
                return -0.377100527;
            }
        } else {
            if (BMI < 22.145328500) {
                return 0.429488510;
            } else {
                if (BMI < 22.230987500) {
                    return -0.228315771;
                } else {
                    return -0.004761520;
                }
            }
        }
    }
}

inline double predict_tree_84(double Gender, double Age, double BMI) {
    if (Age < 57.000000000) {
        if (Age < 56.000000000) {
            if (BMI < 23.795360600) {
                if (BMI < 23.711845400) {
                    return -0.004508634;
                } else {
                    return 0.582688570;
                }
            } else {
                if (BMI < 24.017253900) {
                    return -0.233935982;
                } else {
                    return -0.020127682;
                }
            }
        } else {
            if (BMI < 22.491350200) {
                if (BMI < 22.230987500) {
                    return -0.449496061;
                } else {
                    return 1.777293680;
                }
            } else {
                if (BMI < 24.670375800) {
                    return -0.686723232;
                } else {
                    return -0.098805845;
                }
            }
        }
    } else {
        if (BMI < 23.533042900) {
            if (Age < 60.000000000) {
                if (Age < 58.000000000) {
                    return -0.373636633;
                } else {
                    return 0.940941572;
                }
            } else {
                if (BMI < 23.046875000) {
                    return -0.106850289;
                } else {
                    return 0.420107365;
                }
            }
        } else {
            if (BMI < 23.833004000) {
                if (Age < 58.000000000) {
                    return 1.028302790;
                } else {
                    return -0.397444606;
                }
            } else {
                if (BMI < 23.922422400) {
                    return 0.869307637;
                } else {
                    return 0.042762537;
                }
            }
        }
    }
}

inline double predict_tree_85(double Gender, double Age, double BMI) {
    if (BMI < 23.423557300) {
        if (Gender < 1.000000000) {
            if (BMI < 22.790329000) {
                if (BMI < 22.720438000) {
                    return -0.418985248;
                } else {
                    return -1.646934270;
                }
            } else {
                if (BMI < 22.837369900) {
                    return 1.386097670;
                } else {
                    return -0.283172250;
                }
            }
        } else {
            if (BMI < 23.374725300) {
                if (BMI < 22.230987500) {
                    return -0.064151093;
                } else {
                    return 0.035992954;
                }
            } else {
                if (BMI < 23.413110700) {
                    return -0.454178214;
                } else {
                    return -0.217371941;
                }
            }
        }
    } else {
        if (BMI < 23.808691000) {
            if (BMI < 23.597003900) {
                if (BMI < 23.507804900) {
                    return 0.218680114;
                } else {
                    return -0.186854571;
                }
            } else {
                if (BMI < 23.634033200) {
                    return 0.608972788;
                } else {
                    return 0.113208145;
                }
            }
        } else {
            if (BMI < 23.833004000) {
                if (BMI < 23.808797800) {
                    return -0.040475231;
                } else {
                    return -0.738085926;
                }
            } else {
                if (BMI < 23.893259000) {
                    return 0.346635848;
                } else {
                    return 0.000714752;
                }
            }
        }
    }
}

inline double predict_tree_86(double Gender, double Age, double BMI) {
    if (Age < 72.000000000) {
        if (BMI < 22.598140700) {
            if (BMI < 22.230987500) {
                if (BMI < 22.212966900) {
                    return 0.020566039;
                } else {
                    return -0.635043621;
                }
            } else {
                if (Age < 69.000000000) {
                    return 0.256261736;
                } else {
                    return -1.096368670;
                }
            }
        } else {
            if (BMI < 22.640865300) {
                if (Age < 64.000000000) {
                    return -0.919661462;
                } else {
                    return 0.141466007;
                }
            } else {
                if (Age < 28.000000000) {
                    return -0.080368966;
                } else {
                    return 0.011122287;
                }
            }
        }
    } else {
        if (BMI < 21.952478400) {
            if (Age < 79.000000000) {
                return 1.762220380;
            } else {
                if (BMI < 21.303949400) {
                    return 0.729084909;
                } else {
                    return -0.450567007;
                }
            }
        } else {
            if (BMI < 26.592670400) {
                if (BMI < 25.484764100) {
                    return 0.170542732;
                } else {
                    return -0.341191173;
                }
            } else {
                if (BMI < 26.880950900) {
                    return 1.060921550;
                } else {
                    return 0.226854354;
                }
            }
        }
    }
}

inline double predict_tree_87(double Gender, double Age, double BMI) {
    if (Age < 41.000000000) {
        if (Gender < 1.000000000) {
            if (Age < 21.000000000) {
                return -0.229095757;
            } else {
                if (Age < 24.000000000) {
                    return 0.007061625;
                } else {
                    return -0.086883768;
                }
            }
        } else {
            if (Age < 29.000000000) {
                if (Age < 27.000000000) {
                    return -0.004155509;
                } else {
                    return -0.104000881;
                }
            } else {
                if (Age < 31.000000000) {
                    return 0.169544861;
                } else {
                    return -0.000239352;
                }
            }
        }
    } else {
        if (Gender < 1.000000000) {
            if (Age < 62.000000000) {
                if (Age < 61.000000000) {
                    return 0.094098702;
                } else {
                    return 0.549705505;
                }
            } else {
                if (Age < 72.000000000) {
                    return -0.028493544;
                } else {
                    return 0.116465144;
                }
            }
        } else {
            if (Age < 42.000000000) {
                return 0.338734418;
            } else {
                if (Age < 73.000000000) {
                    return -0.058742266;
                } else {
                    return 0.117953628;
                }
            }
        }
    }
}

inline double predict_tree_88(double Gender, double Age, double BMI) {
    if (BMI < 27.379665400) {
        if (BMI < 27.173914000) {
            if (BMI < 27.084159900) {
                if (BMI < 24.989587800) {
                    return -0.012020153;
                } else {
                    return 0.041937113;
                }
            } else {
                if (BMI < 27.137414900) {
                    return -0.570212901;
                } else {
                    return -0.047324169;
                }
            }
        } else {
            return 0.507724345;
        }
    } else {
        if (BMI < 27.776929900) {
            if (BMI < 27.472526600) {
                return -0.146936476;
            } else {
                if (BMI < 27.714790300) {
                    return -0.318011522;
                } else {
                    return -0.151917726;
                }
            }
        } else {
            return 0.038945161;
        }
    }
}

inline double predict_tree_89(double Gender, double Age, double BMI) {
    if (BMI < 27.776929900) {
        if (BMI < 27.450605400) {
            if (BMI < 27.137414900) {
                if (BMI < 27.084159900) {
                    return 0.010598461;
                } else {
                    return -0.387374401;
                }
            } else {
                if (BMI < 27.160493900) {
                    return 0.432941884;
                } else {
                    return 0.105312347;
                }
            }
        } else {
            if (BMI < 27.714790300) {
                if (BMI < 27.472526600) {
                    return -0.221620694;
                } else {
                    return -0.081379414;
                }
            } else {
                return -0.272486985;
            }
        }
    } else {
        return 0.312010825;
    }
}

inline double predict_tree_90(double Gender, double Age, double BMI) {
    if (Age < 43.000000000) {
        if (BMI < 26.713068000) {
            if (BMI < 26.196188000) {
                if (Age < 42.000000000) {
                    return -0.033570677;
                } else {
                    return -0.282235920;
                }
            } else {
                if (BMI < 26.234567600) {
                    return 1.115156050;
                } else {
                    return 0.066292517;
                }
            }
        } else {
            if (BMI < 26.880950900) {
                if (BMI < 26.869806300) {
                    return -0.517161548;
                } else {
                    return -1.833867430;
                }
            } else {
                if (Age < 29.000000000) {
                    return 0.678610504;
                } else {
                    return -0.320742309;
                }
            }
        }
    } else {
        if (BMI < 23.808797800) {
            if (Age < 45.000000000) {
                if (BMI < 23.483476600) {
                    return -0.324268043;
                } else {
                    return 0.178768873;
                }
            } else {
                if (BMI < 22.375679000) {
                    return -0.058677796;
                } else {
                    return 0.166719601;
                }
            }
        } else {
            if (BMI < 23.833004000) {
                if (Age < 60.000000000) {
                    return 0.043109916;
                } else {
                    return -1.872372390;
                }
            } else {
                if (Age < 75.000000000) {
                    return 0.002893432;
                } else {
                    return 0.205983371;
                }
            }
        }
    }
}

inline double predict_tree_91(double Gender, double Age, double BMI) {
    if (BMI < 27.776929900) {
        if (BMI < 21.218317000) {
            if (BMI < 20.957170500) {
                if (BMI < 20.504934300) {
                    return 0.245477602;
                } else {
                    return 0.027251706;
                }
            } else {
                if (BMI < 21.096191400) {
                    return 0.426333785;
                } else {
                    return 0.250901669;
                }
            }
        } else {
            if (BMI < 21.469150500) {
                if (BMI < 21.303949400) {
                    return -0.342583001;
                } else {
                    return -0.205739081;
                }
            } else {
                if (BMI < 21.877550100) {
                    return 0.151156530;
                } else {
                    return -0.011620585;
                }
            }
        }
    } else {
        return 0.448968530;
    }
}

inline double predict_tree_92(double Gender, double Age, double BMI) {
    if (BMI < 21.877550100) {
        if (BMI < 20.504934300) {
            return 0.351796895;
        } else {
            if (BMI < 20.796730000) {
                return -0.343560547;
            } else {
                if (BMI < 21.367521300) {
                    return 0.248618424;
                } else {
                    return 0.082686976;
                }
            }
        }
    } else {
        if (BMI < 22.100290300) {
            if (BMI < 21.952478400) {
                if (BMI < 21.913805000) {
                    return -0.383813709;
                } else {
                    return 0.227419958;
                }
            } else {
                return -0.503399134;
            }
        } else {
            if (BMI < 22.145328500) {
                return 0.430171698;
            } else {
                if (BMI < 22.230987500) {
                    return -0.203154191;
                } else {
                    return 0.011119750;
                }
            }
        }
    }
}

inline double predict_tree_93(double Gender, double Age, double BMI) {
    if (Age < 54.000000000) {
        if (Age < 52.000000000) {
            if (Age < 49.000000000) {
                if (Age < 48.000000000) {
                    return -0.015744515;
                } else {
                    return -0.176459417;
                }
            } else {
                if (Age < 51.000000000) {
                    return 0.121854357;
                } else {
                    return 0.061068501;
                }
            }
        } else {
            if (Age < 53.000000000) {
                if (Gender < 1.000000000) {
                    return -0.104447536;
                } else {
                    return -0.364934713;
                }
            } else {
                if (Gender < 1.000000000) {
                    return -0.180840969;
                } else {
                    return 0.043899257;
                }
            }
        }
    } else {
        if (Gender < 1.000000000) {
            if (Age < 55.000000000) {
                return 0.551054776;
            } else {
                if (Age < 56.000000000) {
                    return -0.241189629;
                } else {
                    return 0.109311499;
                }
            }
        } else {
            if (Age < 76.000000000) {
                if (Age < 73.000000000) {
                    return -0.028746717;
                } else {
                    return 0.257402956;
                }
            } else {
                if (Age < 79.000000000) {
                    return -0.159664333;
                } else {
                    return -0.035647370;
                }
            }
        }
    }
}

inline double predict_tree_94(double Gender, double Age, double BMI) {
    if (Age < 43.000000000) {
        if (Gender < 1.000000000) {
            if (Age < 21.000000000) {
                return -0.136150822;
            } else {
                if (Age < 29.000000000) {
                    return -0.026078871;
                } else {
                    return -0.070160553;
                }
            }
        } else {
            if (Age < 42.000000000) {
                if (Age < 41.000000000) {
                    return -0.005005393;
                } else {
                    return 0.267336398;
                }
            } else {
                return -0.409868538;
            }
        }
    } else {
        if (Gender < 1.000000000) {
            if (Age < 44.000000000) {
                return 0.468857944;
            } else {
                if (Age < 47.000000000) {
                    return -0.120948836;
                } else {
                    return 0.065219991;
                }
            }
        } else {
            if (Age < 44.000000000) {
                return -0.225454688;
            } else {
                if (Age < 46.000000000) {
                    return 0.215382800;
                } else {
                    return -0.011874967;
                }
            }
        }
    }
}

inline double predict_tree_95(double Gender, double Age, double BMI) {
    if (BMI < 21.952478400) {
        if (BMI < 21.913805000) {
            if (BMI < 21.877550100) {
                if (BMI < 20.796730000) {
                    return -0.026232829;
                } else {
                    return 0.129928038;
                }
            } else {
                return -0.409268260;
            }
        } else {
            return 0.789250195;
        }
    } else {
        if (BMI < 22.100290300) {
            if (BMI < 22.038566600) {
                if (Gender < 1.000000000) {
                    return -0.615950584;
                } else {
                    return -0.184379444;
                }
            } else {
                return -0.459950060;
            }
        } else {
            if (BMI < 22.145328500) {
                return 0.531561017;
            } else {
                if (BMI < 27.776929900) {
                    return -0.000173956;
                } else {
                    return 0.422388494;
                }
            }
        }
    }
}

inline double predict_tree_96(double Gender, double Age, double BMI) {
    if (Age < 57.000000000) {
        if (Age < 21.000000000) {
            if (Gender < 1.000000000) {
                return -0.141758427;
            } else {
                return -0.070401691;
            }
        } else {
            if (Age < 24.000000000) {
                if (Age < 22.000000000) {
                    return 0.026066370;
                } else {
                    return 0.075871043;
                }
            } else {
                if (Age < 45.000000000) {
                    return -0.034576982;
                } else {
                    return 0.011771969;
                }
            }
        }
    } else {
        if (Age < 72.000000000) {
            if (Age < 70.000000000) {
                if (Age < 60.000000000) {
                    return 0.153778836;
                } else {
                    return 0.054516170;
                }
            } else {
                if (Gender < 1.000000000) {
                    return 0.008262666;
                } else {
                    return -0.295435071;
                }
            }
        } else {
            if (Age < 74.000000000) {
                if (Age < 73.000000000) {
                    return 0.155048937;
                } else {
                    return 0.475671619;
                }
            } else {
                if (Age < 75.000000000) {
                    return -0.111731425;
                } else {
                    return 0.084660359;
                }
            }
        }
    }
}

inline double predict_tree_97(double Gender, double Age, double BMI) {
    if (Age < 47.000000000) {
        if (Gender < 1.000000000) {
            if (Age < 44.000000000) {
                if (Age < 43.000000000) {
                    return -0.047874995;
                } else {
                    return 0.311320961;
                }
            } else {
                if (Age < 45.000000000) {
                    return -0.432639301;
                } else {
                    return -0.169866443;
                }
            }
        } else {
            if (Age < 44.000000000) {
                if (Age < 42.000000000) {
                    return -0.017590374;
                } else {
                    return -0.245140687;
                }
            } else {
                if (Age < 46.000000000) {
                    return 0.177102283;
                } else {
                    return -0.063623033;
                }
            }
        }
    } else {
        if (Age < 48.000000000) {
            if (Gender < 1.000000000) {
                return 0.527501106;
            } else {
                return 0.025354940;
            }
        } else {
            if (Age < 49.000000000) {
                if (Gender < 1.000000000) {
                    return -0.308344752;
                } else {
                    return 0.001284814;
                }
            } else {
                if (Gender < 1.000000000) {
                    return 0.057091083;
                } else {
                    return -0.024603762;
                }
            }
        }
    }
}

inline double predict_tree_98(double Gender, double Age, double BMI) {
    if (Age < 57.000000000) {
        if (Age < 55.000000000) {
            if (Age < 33.000000000) {
                if (Age < 23.000000000) {
                    return 0.001969318;
                } else {
                    return -0.046575129;
                }
            } else {
                if (Gender < 1.000000000) {
                    return 0.050200045;
                } else {
                    return -0.010252363;
                }
            }
        } else {
            if (Gender < 1.000000000) {
                if (Age < 56.000000000) {
                    return -0.507437289;
                } else {
                    return 0.021888034;
                }
            } else {
                if (Age < 56.000000000) {
                    return 0.142113209;
                } else {
                    return -0.271491855;
                }
            }
        }
    } else {
        if (Gender < 1.000000000) {
            if (Age < 77.000000000) {
                if (Age < 74.000000000) {
                    return 0.107341029;
                } else {
                    return -0.114775233;
                }
            } else {
                if (Age < 78.000000000) {
                    return 0.413160801;
                } else {
                    return 0.160209268;
                }
            }
        } else {
            if (Age < 60.000000000) {
                if (Age < 58.000000000) {
                    return 0.016830405;
                } else {
                    return 0.269271970;
                }
            } else {
                if (Age < 61.000000000) {
                    return -0.366567224;
                } else {
                    return 0.023175457;
                }
            }
        }
    }
}

inline double predict_tree_99(double Gender, double Age, double BMI) {
    if (Age < 61.000000000) {
        if (BMI < 27.776929900) {
            if (BMI < 26.595745100) {
                if (BMI < 26.592670400) {
                    return -0.017014289;
                } else {
                    return 0.658401787;
                }
            } else {
                if (BMI < 26.851852400) {
                    return -0.511261225;
                } else {
                    return -0.064437054;
                }
            }
        } else {
            if (Age < 40.000000000) {
                if (Age < 29.000000000) {
                    return 0.277234674;
                } else {
                    return -1.107677340;
                }
            } else {
                if (Age < 53.000000000) {
                    return 1.670334100;
                } else {
                    return -0.058434542;
                }
            }
        }
    } else {
        if (BMI < 26.979530300) {
            if (BMI < 26.595745100) {
                if (BMI < 26.282564200) {
                    return 0.102800556;
                } else {
                    return -0.266068369;
                }
            } else {
                if (Age < 74.000000000) {
                    return 0.861483276;
                } else {
                    return -0.184934422;
                }
            }
        } else {
            if (Age < 77.000000000) {
                if (Age < 71.000000000) {
                    return -0.086408839;
                } else {
                    return -0.657017887;
                }
            } else {
                if (BMI < 27.137414900) {
                    return -0.053998385;
                } else {
                    return 0.960569918;
                }
            }
        }
    }
}

inline double predict_tree_100(double Gender, double Age, double BMI) {
    if (Age < 34.000000000) {
        if (BMI < 26.196188000) {
            if (BMI < 24.997549100) {
                if (BMI < 24.618103000) {
                    return -0.059708402;
                } else {
                    return 0.183034450;
                }
            } else {
                if (Age < 31.000000000) {
                    return -0.070005193;
                } else {
                    return -0.409403533;
                }
            }
        } else {
            if (BMI < 26.234567600) {
                if (Age < 24.000000000) {
                    return -0.053210206;
                } else {
                    return 1.134291290;
                }
            } else {
                if (Age < 29.000000000) {
                    return 0.281519502;
                } else {
                    return -0.204141796;
                }
            }
        }
    } else {
        if (BMI < 24.956596400) {
            if (BMI < 24.898143800) {
                if (Age < 70.000000000) {
                    return -0.005802466;
                } else {
                    return 0.149354801;
                }
            } else {
                if (Age < 48.000000000) {
                    return -0.124403685;
                } else {
                    return -1.012914300;
                }
            }
        } else {
            if (BMI < 25.057359700) {
                if (BMI < 25.029760400) {
                    return 0.318023115;
                } else {
                    return 1.163406250;
                }
            } else {
                if (Age < 36.000000000) {
                    return -0.292143017;
                } else {
                    return 0.077807739;
                }
            }
        }
    }
}

inline double predict_tree_101(double Gender, double Age, double BMI) {
    if (Age < 23.000000000) {
        if (Gender < 1.000000000) {
            if (Age < 22.000000000) {
                if (Age < 21.000000000) {
                    return -0.100323133;
                } else {
                    return -0.003826145;
                }
            } else {
                return -0.162255511;
            }
        } else {
            if (Age < 22.000000000) {
                return -0.063002855;
            } else {
                return 0.032265272;
            }
        }
    } else {
        if (Age < 72.000000000) {
            if (Age < 71.000000000) {
                if (Age < 64.000000000) {
                    return -0.002442134;
                } else {
                    return 0.063630015;
                }
            } else {
                return -0.303843915;
            }
        } else {
            if (Gender < 1.000000000) {
                if (Age < 73.000000000) {
                    return 0.383070976;
                } else {
                    return 0.101374365;
                }
            } else {
                if (Age < 75.000000000) {
                    return 0.148246974;
                } else {
                    return -0.120640896;
                }
            }
        }
    }
}

inline double predict_tree_102(double Gender, double Age, double BMI) {
    if (BMI < 21.877550100) {
        if (BMI < 21.545091600) {
            if (BMI < 21.367521300) {
                if (BMI < 20.957170500) {
                    return -0.024896778;
                } else {
                    return 0.216005698;
                }
            } else {
                return -0.413549691;
            }
        } else {
            if (BMI < 21.612812000) {
                return 0.595272303;
            } else {
                if (BMI < 21.718065300) {
                    return 0.106247053;
                } else {
                    return 0.237781137;
                }
            }
        }
    } else {
        if (BMI < 21.913805000) {
            return -0.526790559;
        } else {
            if (BMI < 21.952478400) {
                return 0.536688507;
            } else {
                if (BMI < 22.230987500) {
                    return -0.189130470;
                } else {
                    return -0.011249765;
                }
            }
        }
    }
}

inline double predict_tree_103(double Gender, double Age, double BMI) {
    if (Age < 49.000000000) {
        if (Age < 48.000000000) {
            if (Age < 47.000000000) {
                if (Gender < 1.000000000) {
                    return -0.048446517;
                } else {
                    return 0.005082150;
                }
            } else {
                if (Gender < 1.000000000) {
                    return 0.350618482;
                } else {
                    return 0.046774983;
                }
            }
        } else {
            if (Gender < 1.000000000) {
                return -0.265513122;
            } else {
                return -0.118823774;
            }
        }
    } else {
        if (Age < 50.000000000) {
            if (Gender < 1.000000000) {
                return 0.469702721;
            } else {
                return 0.182139292;
            }
        } else {
            if (Gender < 1.000000000) {
                if (Age < 77.000000000) {
                    return 0.045783509;
                } else {
                    return 0.293480456;
                }
            } else {
                if (Age < 75.000000000) {
                    return 0.002382508;
                } else {
                    return -0.153934374;
                }
            }
        }
    }
}

inline double predict_tree_104(double Gender, double Age, double BMI) {
    if (BMI < 27.776929900) {
        if (BMI < 21.773841900) {
            if (Age < 53.000000000) {
                if (Age < 52.000000000) {
                    return 0.049412232;
                } else {
                    return -0.900948465;
                }
            } else {
                if (BMI < 21.718065300) {
                    return 0.062826797;
                } else {
                    return 2.420292380;
                }
            }
        } else {
            if (BMI < 22.230987500) {
                if (Age < 68.000000000) {
                    return -0.201376408;
                } else {
                    return 0.404505372;
                }
            } else {
                if (BMI < 22.347782100) {
                    return 0.199274063;
                } else {
                    return -0.017731413;
                }
            }
        }
    } else {
        if (Age < 61.000000000) {
            if (Age < 57.000000000) {
                if (Age < 53.000000000) {
                    return 1.218125820;
                } else {
                    return -1.546626090;
                }
            } else {
                return 2.442513940;
            }
        } else {
            if (Age < 67.000000000) {
                return -1.237695930;
            } else {
                if (Age < 71.000000000) {
                    return 1.296564100;
                } else {
                    return -0.040768083;
                }
            }
        }
    }
}

inline double predict_tree_105(double Gender, double Age, double BMI) {
    if (BMI < 26.423570600) {
        if (BMI < 26.312810900) {
            if (Age < 34.000000000) {
                if (BMI < 22.862367600) {
                    return 0.043980785;
                } else {
                    return -0.063002445;
                }
            } else {
                if (BMI < 24.956596400) {
                    return 0.004010910;
                } else {
                    return 0.118021257;
                }
            }
        } else {
            if (Age < 74.000000000) {
                if (Age < 65.000000000) {
                    return 0.309746742;
                } else {
                    return 1.313466790;
                }
            } else {
                if (Age < 77.000000000) {
                    return -1.362190250;
                } else {
                    return 0.679410100;
                }
            }
        }
    } else {
        if (Age < 31.000000000) {
            if (BMI < 26.827421200) {
                if (Age < 27.000000000) {
                    return 0.197929382;
                } else {
                    return 1.290797710;
                }
            } else {
                if (BMI < 26.880950900) {
                    return -0.883952320;
                } else {
                    return 0.363720208;
                }
            }
        } else {
            if (BMI < 26.485540400) {
                if (Age < 64.000000000) {
                    return -1.210131530;
                } else {
                    return -0.192642421;
                }
            } else {
                if (Age < 39.000000000) {
                    return -0.582036674;
                } else {
                    return -0.006250275;
                }
            }
        }
    }
}

inline double predict_tree_106(double Gender, double Age, double BMI) {
    if (BMI < 24.989587800) {
        if (BMI < 24.930748000) {
            if (BMI < 21.773841900) {
                if (BMI < 21.545091600) {
                    return -0.045484491;
                } else {
                    return 0.297859251;
                }
            } else {
                if (BMI < 22.100290300) {
                    return -0.224872723;
                } else {
                    return -0.031133225;
                }
            }
        } else {
            if (BMI < 24.956596400) {
                if (Gender < 1.000000000) {
                    return -0.730135739;
                } else {
                    return -0.221907690;
                }
            } else {
                if (BMI < 24.976087600) {
                    return -0.074845843;
                } else {
                    return -0.440094620;
                }
            }
        }
    } else {
        if (BMI < 25.057359700) {
            if (BMI < 24.997549100) {
                if (Gender < 1.000000000) {
                    return 1.133190040;
                } else {
                    return 0.096054450;
                }
            } else {
                if (BMI < 25.029760400) {
                    return 0.227206469;
                } else {
                    return 0.555874407;
                }
            }
        } else {
            if (Gender < 1.000000000) {
                if (BMI < 25.280109400) {
                    return 0.150112316;
                } else {
                    return -0.020810690;
                }
            } else {
                if (BMI < 25.661151900) {
                    return -0.306256294;
                } else {
                    return 0.165890023;
                }
            }
        }
    }
}

inline double predict_tree_107(double Gender, double Age, double BMI) {
    if (BMI < 21.829952200) {
        if (Age < 71.000000000) {
            if (Age < 22.000000000) {
                if (BMI < 20.504934300) {
                    return 0.210383222;
                } else {
                    return -0.282043725;
                }
            } else {
                if (Age < 28.000000000) {
                    return 0.386306137;
                } else {
                    return -0.016656376;
                }
            }
        } else {
            if (Age < 79.000000000) {
                return 1.335010290;
            } else {
                if (BMI < 21.303949400) {
                    return 0.718349338;
                } else {
                    return -0.446488500;
                }
            }
        }
    } else {
        if (BMI < 22.282478300) {
            if (Age < 39.000000000) {
                if (Age < 29.000000000) {
                    return -0.220587075;
                } else {
                    return 0.174848646;
                }
            } else {
                if (Age < 52.000000000) {
                    return -0.488117278;
                } else {
                    return -0.054311871;
                }
            }
        } else {
            if (BMI < 22.347782100) {
                if (Age < 45.000000000) {
                    return 0.633306503;
                } else {
                    return -0.455757886;
                }
            } else {
                if (BMI < 24.271844900) {
                    return 0.042691115;
                } else {
                    return -0.005074958;
                }
            }
        }
    }
}

inline double predict_tree_108(double Gender, double Age, double BMI) {
    if (Age < 32.000000000) {
        if (BMI < 21.989893000) {
            if (Age < 31.000000000) {
                if (BMI < 21.829952200) {
                    return 0.077238172;
                } else {
                    return 0.396030426;
                }
            } else {
                if (BMI < 21.218317000) {
                    return 0.609242737;
                } else {
                    return -0.899241030;
                }
            }
        } else {
            if (BMI < 22.145328500) {
                if (Age < 29.000000000) {
                    return -0.651246190;
                } else {
                    return -0.051712196;
                }
            } else {
                if (BMI < 26.312810900) {
                    return -0.067591593;
                } else {
                    return 0.156677753;
                }
            }
        }
    } else {
        if (BMI < 26.315052000) {
            if (BMI < 26.312810900) {
                if (BMI < 26.282564200) {
                    return 0.032007094;
                } else {
                    return -0.227421492;
                }
            } else {
                if (Age < 74.000000000) {
                    return 1.099633100;
                } else {
                    return -0.505707085;
                }
            }
        } else {
            if (Age < 39.000000000) {
                if (Age < 37.000000000) {
                    return -0.120509692;
                } else {
                    return -1.099556680;
                }
            } else {
                if (BMI < 26.528511000) {
                    return -0.358913273;
                } else {
                    return 0.049367581;
                }
            }
        }
    }
}

inline double predict_tree_109(double Gender, double Age, double BMI) {
    if (BMI < 26.595745100) {
        if (BMI < 26.312810900) {
            if (BMI < 26.282564200) {
                if (BMI < 25.767219500) {
                    return 0.008937084;
                } else {
                    return 0.098287702;
                }
            } else {
                if (BMI < 26.296566000) {
                    return -0.565111399;
                } else {
                    return -0.156502917;
                }
            }
        } else {
            if (BMI < 26.315052000) {
                if (Gender < 1.000000000) {
                    return 0.864239812;
                } else {
                    return -0.262013435;
                }
            } else {
                if (BMI < 26.555965400) {
                    return 0.028124157;
                } else {
                    return 0.211342201;
                }
            }
        }
    } else {
        if (BMI < 26.713068000) {
            if (Gender < 1.000000000) {
                return -0.505223989;
            } else {
                return 0.545184374;
            }
        } else {
            if (BMI < 27.776929900) {
                if (BMI < 27.450605400) {
                    return -0.022109946;
                } else {
                    return -0.281314284;
                }
            } else {
                return 0.213292301;
            }
        }
    }
}

inline double predict_tree_110(double Gender, double Age, double BMI) {
    if (Age < 22.000000000) {
        if (Gender < 1.000000000) {
            return -0.121095233;
        } else {
            if (Age < 21.000000000) {
                return 0.002491200;
            } else {
                return 0.017466663;
            }
        }
    } else {
        if (Age < 24.000000000) {
            if (Gender < 1.000000000) {
                if (Age < 23.000000000) {
                    return -0.048385523;
                } else {
                    return 0.159964904;
                }
            } else {
                if (Age < 23.000000000) {
                    return 0.186876222;
                } else {
                    return -0.057095397;
                }
            }
        } else {
            if (Age < 57.000000000) {
                if (Age < 56.000000000) {
                    return 0.002541369;
                } else {
                    return -0.374405175;
                }
            } else {
                if (Age < 60.000000000) {
                    return 0.135798380;
                } else {
                    return 0.014918710;
                }
            }
        }
    }
}

inline double predict_tree_111(double Gender, double Age, double BMI) {
    if (BMI < 25.306932400) {
        if (BMI < 24.989587800) {
            if (BMI < 24.976087600) {
                if (BMI < 24.967113500) {
                    return 0.002553622;
                } else {
                    return 0.484674752;
                }
            } else {
                return -0.408720195;
            }
        } else {
            if (BMI < 25.057359700) {
                if (BMI < 24.997549100) {
                    return 0.983241737;
                } else {
                    return 0.403953046;
                }
            } else {
                if (BMI < 25.207756000) {
                    return -0.098576494;
                } else {
                    return 0.259740323;
                }
            }
        }
    } else {
        if (BMI < 25.333334000) {
            return -0.661845148;
        } else {
            if (BMI < 25.592210800) {
                if (Gender < 1.000000000) {
                    return -0.048363242;
                } else {
                    return -0.306112766;
                }
            } else {
                if (BMI < 25.725517300) {
                    return 0.166949019;
                } else {
                    return -0.003544926;
                }
            }
        }
    }
}

inline double predict_tree_112(double Gender, double Age, double BMI) {
    if (Age < 21.000000000) {
        return -0.130167350;
    } else {
        if (Age < 72.000000000) {
            if (Age < 71.000000000) {
                if (Age < 64.000000000) {
                    return 0.003234490;
                } else {
                    return 0.055501409;
                }
            } else {
                if (Gender < 1.000000000) {
                    return -0.129402980;
                } else {
                    return -0.235503122;
                }
            }
        } else {
            if (Age < 74.000000000) {
                if (Gender < 1.000000000) {
                    return 0.280001998;
                } else {
                    return 0.124358244;
                }
            } else {
                if (Age < 76.000000000) {
                    return 0.078704014;
                } else {
                    return -0.040326461;
                }
            }
        }
    }
}

inline double predict_tree_113(double Gender, double Age, double BMI) {
    if (Age < 29.000000000) {
        if (BMI < 26.880950900) {
            if (BMI < 26.851852400) {
                if (BMI < 26.196188000) {
                    return -0.040969592;
                } else {
                    return 0.239791825;
                }
            } else {
                return -1.117656950;
            }
        } else {
            if (Age < 27.000000000) {
                if (BMI < 27.160493900) {
                    return 0.259550571;
                } else {
                    return -1.058426860;
                }
            } else {
                if (BMI < 27.472526600) {
                    return 1.696167590;
                } else {
                    return 0.092141166;
                }
            }
        }
    } else {
        if (BMI < 21.096191400) {
            if (Age < 34.000000000) {
                if (Age < 33.000000000) {
                    return 0.210201845;
                } else {
                    return -1.096632480;
                }
            } else {
                if (Age < 50.000000000) {
                    return 0.620958686;
                } else {
                    return 0.053008828;
                }
            }
        } else {
            if (BMI < 21.718065300) {
                if (Age < 71.000000000) {
                    return -0.285625368;
                } else {
                    return 0.913363993;
                }
            } else {
                if (BMI < 21.829952200) {
                    return 0.521513224;
                } else {
                    return 0.021452464;
                }
            }
        }
    }
}

inline double predict_tree_114(double Gender, double Age, double BMI) {
    if (Age < 22.000000000) {
        if (Gender < 1.000000000) {
            if (Age < 21.000000000) {
                return -0.148144633;
            } else {
                return -0.028648280;
            }
        } else {
            if (Age < 21.000000000) {
                return -0.024284720;
            } else {
                return -0.101420313;
            }
        }
    } else {
        if (Age < 23.000000000) {
            if (Gender < 1.000000000) {
                return 0.008213416;
            } else {
                return 0.245239362;
            }
        } else {
            if (Gender < 1.000000000) {
                if (Age < 32.000000000) {
                    return -0.032904282;
                } else {
                    return 0.049640860;
                }
            } else {
                if (Age < 56.000000000) {
                    return 0.009723858;
                } else {
                    return -0.085476622;
                }
            }
        }
    }
}

inline double predict_tree_115(double Gender, double Age, double BMI) {
    if (Age < 28.000000000) {
        if (Age < 24.000000000) {
            if (Age < 22.000000000) {
                if (Age < 21.000000000) {
                    return -0.071595922;
                } else {
                    return -0.040487543;
                }
            } else {
                if (Age < 23.000000000) {
                    return 0.062084131;
                } else {
                    return -0.036512218;
                }
            }
        } else {
            if (Gender < 1.000000000) {
                if (Age < 25.000000000) {
                    return -0.261669040;
                } else {
                    return -0.076823689;
                }
            } else {
                if (Age < 26.000000000) {
                    return -0.112997621;
                } else {
                    return -0.031143371;
                }
            }
        }
    } else {
        if (Age < 64.000000000) {
            if (Age < 60.000000000) {
                if (Age < 49.000000000) {
                    return -0.004532123;
                } else {
                    return 0.041059397;
                }
            } else {
                if (Gender < 1.000000000) {
                    return -0.039572213;
                } else {
                    return -0.158547327;
                }
            }
        } else {
            if (Gender < 1.000000000) {
                if (Age < 67.000000000) {
                    return -0.063416392;
                } else {
                    return 0.117356434;
                }
            } else {
                if (Age < 65.000000000) {
                    return 0.356062144;
                } else {
                    return -0.017867358;
                }
            }
        }
    }
}

inline double predict_tree_116(double Gender, double Age, double BMI) {
    if (BMI < 27.450605400) {
        if (BMI < 27.173914000) {
            if (BMI < 27.160493900) {
                if (BMI < 21.218317000) {
                    return 0.218272209;
                } else {
                    return 0.007175327;
                }
            } else {
                return -0.467479378;
            }
        } else {
            if (BMI < 27.379665400) {
                return 0.642991185;
            } else {
                return 0.336727172;
            }
        }
    } else {
        if (BMI < 27.776929900) {
            if (BMI < 27.714790300) {
                return -0.287809759;
            } else {
                return -0.146436051;
            }
        } else {
            return -0.687933803;
        }
    }
}

inline double predict_tree_117(double Gender, double Age, double BMI) {
    if (Age < 28.000000000) {
        if (Age < 23.000000000) {
            if (Age < 22.000000000) {
                if (Gender < 1.000000000) {
                    return -0.044782866;
                } else {
                    return -0.006409299;
                }
            } else {
                if (Gender < 1.000000000) {
                    return 0.059018772;
                } else {
                    return 0.086258046;
                }
            }
        } else {
            if (Age < 25.000000000) {
                if (Gender < 1.000000000) {
                    return -0.115323000;
                } else {
                    return -0.054130759;
                }
            } else {
                if (Gender < 1.000000000) {
                    return -0.029687157;
                } else {
                    return -0.047580823;
                }
            }
        }
    } else {
        if (Age < 54.000000000) {
            if (Age < 51.000000000) {
                if (Age < 47.000000000) {
                    return 0.010266094;
                } else {
                    return 0.097634181;
                }
            } else {
                if (Age < 53.000000000) {
                    return -0.160106584;
                } else {
                    return 0.006243319;
                }
            }
        } else {
            if (Age < 55.000000000) {
                if (Gender < 1.000000000) {
                    return 0.741111875;
                } else {
                    return -0.116208941;
                }
            } else {
                if (Age < 65.000000000) {
                    return 0.086525798;
                } else {
                    return -0.004726950;
                }
            }
        }
    }
}

inline double predict_tree_118(double Gender, double Age, double BMI) {
    if (BMI < 25.992439300) {
        if (BMI < 25.767219500) {
            if (BMI < 25.762678100) {
                if (BMI < 25.755157500) {
                    return 0.004386003;
                } else {
                    return 0.680685878;
                }
            } else {
                if (Gender < 1.000000000) {
                    return -0.604884505;
                } else {
                    return 0.636076331;
                }
            }
        } else {
            if (Gender < 1.000000000) {
                if (BMI < 25.772504800) {
                    return 0.645282745;
                } else {
                    return 0.265842408;
                }
            } else {
                if (BMI < 25.795917500) {
                    return 1.050294880;
                } else {
                    return -0.466271728;
                }
            }
        }
    } else {
        if (BMI < 26.023048400) {
            if (BMI < 26.014568300) {
                return -0.281363070;
            } else {
                return -1.368666410;
            }
        } else {
            if (Gender < 1.000000000) {
                if (BMI < 27.776929900) {
                    return -0.036044352;
                } else {
                    return 0.281667709;
                }
            } else {
                if (BMI < 26.078971900) {
                    return 1.131499410;
                } else {
                    return 0.190895349;
                }
            }
        }
    }
}

inline double predict_tree_119(double Gender, double Age, double BMI) {
    if (Age < 28.000000000) {
        if (Age < 27.000000000) {
            if (Age < 21.000000000) {
                if (Gender < 1.000000000) {
                    return -0.075345226;
                } else {
                    return -0.042020187;
                }
            } else {
                if (Age < 24.000000000) {
                    return 0.021001784;
                } else {
                    return -0.034520540;
                }
            }
        } else {
            if (Gender < 1.000000000) {
                return -0.021212203;
            } else {
                return -0.180862531;
            }
        }
    } else {
        if (Age < 38.000000000) {
            if (Gender < 1.000000000) {
                if (Age < 29.000000000) {
                    return 0.218337536;
                } else {
                    return 0.010103996;
                }
            } else {
                if (Age < 37.000000000) {
                    return 0.074016012;
                } else {
                    return 0.224622399;
                }
            }
        } else {
            if (Age < 54.000000000) {
                if (Age < 52.000000000) {
                    return -0.009766534;
                } else {
                    return -0.124235056;
                }
            } else {
                if (Age < 55.000000000) {
                    return 0.260433167;
                } else {
                    return 0.027257785;
                }
            }
        }
    }
}

inline double predict_tree_120(double Gender, double Age, double BMI) {
    if (BMI < 27.776929900) {
        if (BMI < 27.450605400) {
            if (BMI < 27.173914000) {
                if (BMI < 21.829952200) {
                    return 0.092328340;
                } else {
                    return -0.002315517;
                }
            } else {
                if (Age < 65.000000000) {
                    return -0.178823888;
                } else {
                    return 1.325307250;
                }
            }
        } else {
            if (Age < 77.000000000) {
                if (Age < 71.000000000) {
                    return -0.144511923;
                } else {
                    return -0.986142933;
                }
            } else {
                if (BMI < 27.472526600) {
                    return 1.131944660;
                } else {
                    return 0.329454362;
                }
            }
        }
    } else {
        if (Age < 72.000000000) {
            if (Age < 53.000000000) {
                if (Age < 36.000000000) {
                    return -0.320702583;
                } else {
                    return 1.430798770;
                }
            } else {
                if (Age < 57.000000000) {
                    return -1.514067770;
                } else {
                    return -0.100061595;
                }
            }
        } else {
            return 1.656754970;
        }
    }
}

inline double predict_tree_121(double Gender, double Age, double BMI) {
    if (BMI < 23.437500000) {
        if (Age < 75.000000000) {
            if (Age < 70.000000000) {
                if (BMI < 23.147254900) {
                    return -0.009915049;
                } else {
                    return -0.183048725;
                }
            } else {
                if (BMI < 23.374725300) {
                    return 0.838888466;
                } else {
                    return -0.477321714;
                }
            }
        } else {
            if (BMI < 22.829963700) {
                if (Age < 78.000000000) {
                    return -1.497602340;
                } else {
                    return -0.111204632;
                }
            } else {
                if (Age < 78.000000000) {
                    return 0.212879613;
                } else {
                    return -0.883972406;
                }
            }
        }
    } else {
        if (BMI < 23.634033200) {
            if (BMI < 23.597003900) {
                if (BMI < 23.555555300) {
                    return 0.178225398;
                } else {
                    return -0.358599186;
                }
            } else {
                if (Age < 70.000000000) {
                    return 0.857162774;
                } else {
                    return -0.679125071;
                }
            }
        } else {
            if (BMI < 23.711845400) {
                if (Age < 24.000000000) {
                    return 0.294961184;
                } else {
                    return -0.551992357;
                }
            } else {
                if (BMI < 23.795360600) {
                    return 0.309520274;
                } else {
                    return 0.008404810;
                }
            }
        }
    }
}

inline double predict_tree_122(double Gender, double Age, double BMI) {
    if (BMI < 27.379665400) {
        if (BMI < 27.173914000) {
            if (BMI < 25.306932400) {
                if (BMI < 24.989587800) {
                    return -0.008227246;
                } else {
                    return 0.120869674;
                }
            } else {
                if (BMI < 25.333334000) {
                    return -0.609060585;
                } else {
                    return -0.037466779;
                }
            }
        } else {
            return 0.692353964;
        }
    } else {
        if (BMI < 27.472526600) {
            if (BMI < 27.450605400) {
                return -0.135392964;
            } else {
                return -0.648494720;
            }
        } else {
            if (BMI < 27.776929900) {
                if (BMI < 27.714790300) {
                    return -0.079098903;
                } else {
                    return -0.176281348;
                }
            } else {
                return -0.030525550;
            }
        }
    }
}

inline double predict_tree_123(double Gender, double Age, double BMI) {
    if (BMI < 22.282478300) {
        if (Age < 53.000000000) {
            if (Age < 22.000000000) {
                if (BMI < 22.060354200) {
                    return -0.258370072;
                } else {
                    return -0.756059706;
                }
            } else {
                if (Age < 28.000000000) {
                    return 0.055024989;
                } else {
                    return -0.173041075;
                }
            }
        } else {
            if (Age < 54.000000000) {
                if (BMI < 22.100290300) {
                    return 0.519830644;
                } else {
                    return 1.560606600;
                }
            } else {
                if (BMI < 22.145328500) {
                    return 0.221852541;
                } else {
                    return -0.444848627;
                }
            }
        }
    } else {
        if (BMI < 22.347782100) {
            if (Age < 45.000000000) {
                if (Age < 35.000000000) {
                    return 0.135493904;
                } else {
                    return 1.825178380;
                }
            } else {
                if (Age < 49.000000000) {
                    return -1.067871090;
                } else {
                    return 0.194203585;
                }
            }
        } else {
            if (BMI < 25.767219500) {
                if (BMI < 25.762678100) {
                    return -0.010218432;
                } else {
                    return -0.621762514;
                }
            } else {
                if (BMI < 25.992439300) {
                    return 0.295903891;
                } else {
                    return -0.001289741;
                }
            }
        }
    }
}

inline double predict_tree_124(double Gender, double Age, double BMI) {
    if (BMI < 27.776929900) {
        if (BMI < 27.379665400) {
            if (BMI < 27.173914000) {
                if (BMI < 23.225431400) {
                    return -0.029057367;
                } else {
                    return 0.017170731;
                }
            } else {
                return 0.382003963;
            }
        } else {
            if (BMI < 27.714790300) {
                if (BMI < 27.450605400) {
                    return -0.351920128;
                } else {
                    return -0.236654982;
                }
            } else {
                return -0.108911216;
            }
        }
    } else {
        return 0.445137858;
    }
}

inline double predict_tree_125(double Gender, double Age, double BMI) {
    if (BMI < 24.304616900) {
        if (Age < 69.000000000) {
            if (BMI < 24.221452700) {
                if (BMI < 23.893259000) {
                    return 0.014904002;
                } else {
                    return -0.137394086;
                }
            } else {
                if (Age < 28.000000000) {
                    return -0.326890320;
                } else {
                    return 0.394367218;
                }
            }
        } else {
            if (BMI < 22.829963700) {
                if (BMI < 22.230987500) {
                    return 0.447504669;
                } else {
                    return -0.653558731;
                }
            } else {
                if (BMI < 24.056934400) {
                    return 0.306776255;
                } else {
                    return 1.044727560;
                }
            }
        }
    } else {
        if (BMI < 26.196188000) {
            if (Age < 78.000000000) {
                if (Age < 70.000000000) {
                    return -0.048462514;
                } else {
                    return -0.339029640;
                }
            } else {
                if (BMI < 25.762678100) {
                    return 0.635201752;
                } else {
                    return -0.939749181;
                }
            }
        } else {
            if (Age < 29.000000000) {
                if (Age < 25.000000000) {
                    return 0.024993941;
                } else {
                    return 0.707561195;
                }
            } else {
                if (Age < 39.000000000) {
                    return -0.170101121;
                } else {
                    return 0.044806711;
                }
            }
        }
    }
}

inline double predict_tree_126(double Gender, double Age, double BMI) {
    if (BMI < 21.218317000) {
        if (BMI < 21.096191400) {
            if (BMI < 20.504934300) {
                return 0.218538284;
            } else {
                if (BMI < 20.796730000) {
                    return -0.115260787;
                } else {
                    return 0.176315248;
                }
            }
        } else {
            return 0.461061805;
        }
    } else {
        if (BMI < 27.379665400) {
            if (BMI < 27.173914000) {
                if (BMI < 27.084159900) {
                    return -0.005521837;
                } else {
                    return -0.200397894;
                }
            } else {
                return 0.405974746;
            }
        } else {
            if (BMI < 27.472526600) {
                if (BMI < 27.450605400) {
                    return -0.143016189;
                } else {
                    return -0.374577314;
                }
            } else {
                if (BMI < 27.714790300) {
                    return 0.028704422;
                } else {
                    return -0.220070988;
                }
            }
        }
    }
}

inline double predict_tree_127(double Gender, double Age, double BMI) {
    if (Age < 63.000000000) {
        if (Age < 58.000000000) {
            if (Age < 57.000000000) {
                if (Age < 56.000000000) {
                    return -0.010794726;
                } else {
                    return -0.157943875;
                }
            } else {
                if (Gender < 1.000000000) {
                    return 0.303428173;
                } else {
                    return 0.071686119;
                }
            }
        } else {
            if (Gender < 1.000000000) {
                if (Age < 61.000000000) {
                    return -0.146734715;
                } else {
                    return 0.120872721;
                }
            } else {
                if (Age < 59.000000000) {
                    return 0.124115467;
                } else {
                    return -0.202581346;
                }
            }
        }
    } else {
        if (Age < 69.000000000) {
            if (Gender < 1.000000000) {
                if (Age < 67.000000000) {
                    return 0.091397226;
                } else {
                    return 0.382761061;
                }
            } else {
                if (Age < 67.000000000) {
                    return 0.160778373;
                } else {
                    return -0.045543186;
                }
            }
        } else {
            if (Age < 72.000000000) {
                if (Age < 71.000000000) {
                    return -0.051906392;
                } else {
                    return -0.282647848;
                }
            } else {
                if (Age < 76.000000000) {
                    return 0.119295970;
                } else {
                    return 0.015776932;
                }
            }
        }
    }
}

inline double predict_tree_128(double Gender, double Age, double BMI) {
    if (Age < 32.000000000) {
        if (Gender < 1.000000000) {
            if (Age < 22.000000000) {
                return -0.138345852;
            } else {
                if (Age < 26.000000000) {
                    return 0.014622600;
                } else {
                    return -0.082512073;
                }
            }
        } else {
            if (Age < 21.000000000) {
                return 0.136488497;
            } else {
                if (Age < 29.000000000) {
                    return -0.053146247;
                } else {
                    return 0.031246534;
                }
            }
        }
    } else {
        if (Gender < 1.000000000) {
            if (Age < 74.000000000) {
                if (Age < 72.000000000) {
                    return 0.051720880;
                } else {
                    return 0.367480427;
                }
            } else {
                if (Age < 75.000000000) {
                    return -0.484405965;
                } else {
                    return 0.012469449;
                }
            }
        } else {
            if (Age < 70.000000000) {
                if (Age < 68.000000000) {
                    return -0.009984033;
                } else {
                    return 0.233994976;
                }
            } else {
                if (Age < 73.000000000) {
                    return -0.458279222;
                } else {
                    return 0.001646652;
                }
            }
        }
    }
}

inline double predict_tree_129(double Gender, double Age, double BMI) {
    if (Age < 28.000000000) {
        if (Gender < 1.000000000) {
            if (Age < 21.000000000) {
                return -0.232294932;
            } else {
                if (Age < 27.000000000) {
                    return -0.062933348;
                } else {
                    return 0.036810253;
                }
            }
        } else {
            if (Age < 23.000000000) {
                if (Age < 21.000000000) {
                    return 0.049740553;
                } else {
                    return 0.002708829;
                }
            } else {
                if (Age < 26.000000000) {
                    return -0.131947577;
                } else {
                    return -0.043468688;
                }
            }
        }
    } else {
        if (Age < 73.000000000) {
            if (Age < 70.000000000) {
                if (Age < 64.000000000) {
                    return 0.011623571;
                } else {
                    return 0.082134798;
                }
            } else {
                if (Gender < 1.000000000) {
                    return 0.080167361;
                } else {
                    return -0.265693963;
                }
            }
        } else {
            if (Age < 74.000000000) {
                return 0.249518663;
            } else {
                if (Age < 75.000000000) {
                    return -0.098364338;
                } else {
                    return 0.050341356;
                }
            }
        }
    }
}

inline double predict_tree_130(double Gender, double Age, double BMI) {
    if (BMI < 21.367521300) {
        if (Age < 34.000000000) {
            if (Age < 28.000000000) {
                if (BMI < 21.096191400) {
                    return 0.024196362;
                } else {
                    return 0.625141919;
                }
            } else {
                if (BMI < 21.218317000) {
                    return -0.005844608;
                } else {
                    return -1.136047720;
                }
            }
        } else {
            if (Age < 35.000000000) {
                if (BMI < 20.504934300) {
                    return 0.064969234;
                } else {
                    return 1.595827460;
                }
            } else {
                if (Age < 37.000000000) {
                    return -0.446868807;
                } else {
                    return 0.458337426;
                }
            }
        }
    } else {
        if (BMI < 21.545091600) {
            if (Age < 38.000000000) {
                if (Age < 30.000000000) {
                    return -0.295300931;
                } else {
                    return 0.524941623;
                }
            } else {
                if (Age < 47.000000000) {
                    return -1.354383350;
                } else {
                    return 0.063581631;
                }
            }
        } else {
            if (BMI < 21.773841900) {
                if (Age < 59.000000000) {
                    return 0.194575518;
                } else {
                    return 1.038659450;
                }
            } else {
                if (BMI < 22.230987500) {
                    return -0.141190991;
                } else {
                    return 0.022259721;
                }
            }
        }
    }
}

inline double predict_tree_131(double Gender, double Age, double BMI) {
    if (BMI < 20.796730000) {
        if (Age < 23.000000000) {
            if (Age < 21.000000000) {
                if (BMI < 20.504934300) {
                    return 0.010836007;
                } else {
                    return -0.104936138;
                }
            } else {
                return -1.209318040;
            }
        } else {
            if (Age < 26.000000000) {
                if (Age < 24.000000000) {
                    return -0.108594969;
                } else {
                    return 0.797191083;
                }
            } else {
                if (Age < 34.000000000) {
                    return -0.731498241;
                } else {
                    return -0.221403837;
                }
            }
        }
    } else {
        if (BMI < 21.952478400) {
            if (Age < 71.000000000) {
                if (Age < 56.000000000) {
                    return 0.111320414;
                } else {
                    return -0.318555832;
                }
            } else {
                if (Age < 79.000000000) {
                    return 1.605491640;
                } else {
                    return -0.340771586;
                }
            }
        } else {
            if (BMI < 22.100290300) {
                if (Age < 37.000000000) {
                    return -0.121131659;
                } else {
                    return -0.622255027;
                }
            } else {
                if (BMI < 22.145328500) {
                    return 0.558074594;
                } else {
                    return 0.001342210;
                }
            }
        }
    }
}

inline double predict_tree_132(double Gender, double Age, double BMI) {
    if (BMI < 24.560325600) {
        if (BMI < 24.515596400) {
            if (Age < 69.000000000) {
                if (Age < 67.000000000) {
                    return 0.011880646;
                } else {
                    return -0.364593953;
                }
            } else {
                if (BMI < 22.829963700) {
                    return -0.352351397;
                } else {
                    return 0.307797700;
                }
            }
        } else {
            if (Age < 51.000000000) {
                if (Age < 31.000000000) {
                    return -0.352692127;
                } else {
                    return 1.225483060;
                }
            } else {
                if (Age < 70.000000000) {
                    return -0.815198421;
                } else {
                    return 0.851806641;
                }
            }
        }
    } else {
        if (BMI < 24.618103000) {
            if (BMI < 24.609375000) {
                if (Age < 39.000000000) {
                    return 0.416849941;
                } else {
                    return -0.558387756;
                }
            } else {
                if (Age < 48.000000000) {
                    return -1.366036770;
                } else {
                    return -0.058850672;
                }
            }
        } else {
            if (Age < 61.000000000) {
                if (Age < 59.000000000) {
                    return -0.041280679;
                } else {
                    return -0.372013927;
                }
            } else {
                if (Age < 69.000000000) {
                    return 0.225357935;
                } else {
                    return -0.072683275;
                }
            }
        }
    }
}

inline double predict_tree_133(double Gender, double Age, double BMI) {
    if (BMI < 24.989587800) {
        if (BMI < 24.898143800) {
            if (Age < 70.000000000) {
                if (Age < 67.000000000) {
                    return -0.014887863;
                } else {
                    return -0.347155333;
                }
            } else {
                if (BMI < 24.654832800) {
                    return 0.232002363;
                } else {
                    return -0.261682600;
                }
            }
        } else {
            if (Age < 27.000000000) {
                if (BMI < 24.930748000) {
                    return 0.146380275;
                } else {
                    return -0.913422525;
                }
            } else {
                if (Age < 29.000000000) {
                    return 1.283558010;
                } else {
                    return -0.216490701;
                }
            }
        }
    } else {
        if (BMI < 24.997549100) {
            if (Age < 69.000000000) {
                if (Age < 32.000000000) {
                    return 0.621247292;
                } else {
                    return 1.995113020;
                }
            } else {
                return -0.685920715;
            }
        } else {
            if (BMI < 25.057359700) {
                if (Age < 34.000000000) {
                    return -0.137166291;
                } else {
                    return 0.608795404;
                }
            } else {
                if (Age < 25.000000000) {
                    return -0.132234454;
                } else {
                    return 0.022703998;
                }
            }
        }
    }
}

inline double predict_tree_134(double Gender, double Age, double BMI) {
    if (Age < 28.000000000) {
        if (BMI < 26.196188000) {
            if (BMI < 26.078971900) {
                if (BMI < 26.040952700) {
                    return -0.061102565;
                } else {
                    return 1.552819490;
                }
            } else {
                if (Age < 23.000000000) {
                    return -0.261039466;
                } else {
                    return -1.213545440;
                }
            }
        } else {
            if (Age < 27.000000000) {
                if (BMI < 26.234567600) {
                    return 1.087971090;
                } else {
                    return -0.032451484;
                }
            } else {
                if (BMI < 26.979530300) {
                    return 1.223448630;
                } else {
                    return -0.508261502;
                }
            }
        }
    } else {
        if (BMI < 21.096191400) {
            if (Age < 34.000000000) {
                if (Age < 33.000000000) {
                    return 0.577927411;
                } else {
                    return -1.246554970;
                }
            } else {
                if (Age < 50.000000000) {
                    return 0.673088372;
                } else {
                    return 0.008909702;
                }
            }
        } else {
            if (BMI < 22.100290300) {
                if (Age < 53.000000000) {
                    return -0.243363053;
                } else {
                    return 0.165096968;
                }
            } else {
                if (BMI < 24.569315000) {
                    return 0.062101271;
                } else {
                    return -0.019149035;
                }
            }
        }
    }
}

inline double predict_tree_135(double Gender, double Age, double BMI) {
    if (Age < 57.000000000) {
        if (BMI < 27.776929900) {
            if (BMI < 23.437500000) {
                if (BMI < 23.323417700) {
                    return -0.014856971;
                } else {
                    return -0.382917523;
                }
            } else {
                if (BMI < 23.795360600) {
                    return 0.198786810;
                } else {
                    return -0.010928838;
                }
            }
        } else {
            if (Age < 51.000000000) {
                if (Age < 49.000000000) {
                    return -0.722121477;
                } else {
                    return 0.926884830;
                }
            } else {
                return -1.485131620;
            }
        }
    } else {
        if (Age < 58.000000000) {
            if (BMI < 24.323228800) {
                if (BMI < 23.833004000) {
                    return -0.012226664;
                } else {
                    return -0.947124362;
                }
            } else {
                if (BMI < 26.794937100) {
                    return 0.861593783;
                } else {
                    return -0.083746485;
                }
            }
        } else {
            if (BMI < 23.533042900) {
                if (BMI < 23.456790900) {
                    return 0.120656267;
                } else {
                    return 0.891970575;
                }
            } else {
                if (BMI < 23.833004000) {
                    return -0.562028587;
                } else {
                    return 0.050374787;
                }
            }
        }
    }
}

inline double predict_tree_136(double Gender, double Age, double BMI) {
    if (Age < 61.000000000) {
        if (BMI < 27.776929900) {
            if (BMI < 27.160493900) {
                if (BMI < 26.979530300) {
                    return -0.016682476;
                } else {
                    return 0.418544263;
                }
            } else {
                if (Age < 58.000000000) {
                    return -0.333961755;
                } else {
                    return 0.377797276;
                }
            }
        } else {
            if (Age < 40.000000000) {
                if (Age < 29.000000000) {
                    return 0.264991134;
                } else {
                    return -1.087113740;
                }
            } else {
                if (Age < 57.000000000) {
                    return 0.451942503;
                } else {
                    return 1.611471650;
                }
            }
        }
    } else {
        if (BMI < 23.046875000) {
            if (BMI < 22.145328500) {
                if (Age < 69.000000000) {
                    return -0.045096803;
                } else {
                    return 1.589552880;
                }
            } else {
                if (Age < 75.000000000) {
                    return -0.156281501;
                } else {
                    return -0.929016531;
                }
            }
        } else {
            if (BMI < 23.507804900) {
                if (BMI < 23.456790900) {
                    return 0.344695926;
                } else {
                    return 2.103405710;
                }
            } else {
                if (BMI < 23.833004000) {
                    return -0.415103287;
                } else {
                    return 0.079164714;
                }
            }
        }
    }
}

inline double predict_tree_137(double Gender, double Age, double BMI) {
    if (Age < 42.000000000) {
        if (Age < 41.000000000) {
            if (Age < 40.000000000) {
                if (Age < 39.000000000) {
                    return -0.008678013;
                } else {
                    return 0.167281181;
                }
            } else {
                if (Gender < 1.000000000) {
                    return -0.076905847;
                } else {
                    return -0.343583107;
                }
            }
        } else {
            if (Gender < 1.000000000) {
                return 0.163824111;
            } else {
                return 0.291083664;
            }
        }
    } else {
        if (Age < 43.000000000) {
            if (Gender < 1.000000000) {
                return -0.076152436;
            } else {
                return -0.308706671;
            }
        } else {
            if (Gender < 1.000000000) {
                if (Age < 51.000000000) {
                    return 0.100251436;
                } else {
                    return -0.035764191;
                }
            } else {
                if (Age < 55.000000000) {
                    return -0.082718372;
                } else {
                    return -0.003824066;
                }
            }
        }
    }
}

inline double predict_tree_138(double Gender, double Age, double BMI) {
    if (BMI < 22.230987500) {
        if (Age < 22.000000000) {
            if (BMI < 22.038566600) {
                if (BMI < 21.877550100) {
                    return -0.409073323;
                } else {
                    return 0.454814911;
                }
            } else {
                if (BMI < 22.189348200) {
                    return -1.515281440;
                } else {
                    return -0.117467366;
                }
            }
        } else {
            if (Age < 71.000000000) {
                if (BMI < 22.212966900) {
                    return -0.025975959;
                } else {
                    return -0.558485031;
                }
            } else {
                if (Age < 75.000000000) {
                    return 1.590904470;
                } else {
                    return -0.207888767;
                }
            }
        }
    } else {
        if (BMI < 22.347782100) {
            if (Age < 66.000000000) {
                if (Age < 35.000000000) {
                    return 0.073978186;
                } else {
                    return 0.800347865;
                }
            } else {
                if (BMI < 22.313278200) {
                    return -1.067322490;
                } else {
                    return -0.017321395;
                }
            }
        } else {
            if (BMI < 24.989587800) {
                if (Age < 72.000000000) {
                    return -0.021082753;
                } else {
                    return 0.172960341;
                }
            } else {
                if (BMI < 25.000000000) {
                    return 0.803753018;
                } else {
                    return 0.019474655;
                }
            }
        }
    }
}

inline double predict_tree_139(double Gender, double Age, double BMI) {
    if (BMI < 21.367521300) {
        if (BMI < 21.303949400) {
            if (BMI < 20.504934300) {
                return 0.260989845;
            } else {
                if (BMI < 20.796730000) {
                    return -0.162479326;
                } else {
                    return 0.066395216;
                }
            }
        } else {
            return 0.469666928;
        }
    } else {
        if (BMI < 21.469150500) {
            return -0.439585418;
        } else {
            if (BMI < 25.306932400) {
                if (BMI < 24.989587800) {
                    return 0.000455927;
                } else {
                    return 0.162673056;
                }
            } else {
                if (BMI < 25.333334000) {
                    return -0.475746304;
                } else {
                    return -0.016774599;
                }
            }
        }
    }
}

inline double predict_tree_140(double Gender, double Age, double BMI) {
    if (Age < 73.000000000) {
        if (Age < 70.000000000) {
            if (Age < 64.000000000) {
                if (Age < 62.000000000) {
                    return -0.003706423;
                } else {
                    return -0.207399070;
                }
            } else {
                if (Age < 65.000000000) {
                    return 0.319656491;
                } else {
                    return -0.004702869;
                }
            }
        } else {
            if (Gender < 1.000000000) {
                if (Age < 72.000000000) {
                    return -0.022074239;
                } else {
                    return 0.263862520;
                }
            } else {
                if (Age < 71.000000000) {
                    return -0.430159599;
                } else {
                    return -0.213402480;
                }
            }
        }
    } else {
        if (Age < 74.000000000) {
            if (Gender < 1.000000000) {
                return 0.301874608;
            } else {
                return 0.200321198;
            }
        } else {
            if (Gender < 1.000000000) {
                if (Age < 75.000000000) {
                    return -0.575385034;
                } else {
                    return 0.085018970;
                }
            } else {
                if (Age < 75.000000000) {
                    return 0.616532683;
                } else {
                    return -0.069813252;
                }
            }
        }
    }
}

inline double predict_tree_141(double Gender, double Age, double BMI) {
    if (BMI < 27.379665400) {
        if (BMI < 27.173914000) {
            if (BMI < 26.078971900) {
                if (BMI < 25.992439300) {
                    return 0.006228918;
                } else {
                    return -0.194075063;
                }
            } else {
                if (Age < 76.000000000) {
                    return 0.096845657;
                } else {
                    return -0.501985848;
                }
            }
        } else {
            if (Age < 68.000000000) {
                if (Age < 48.000000000) {
                    return 0.578012466;
                } else {
                    return -0.763629258;
                }
            } else {
                return 1.959532740;
            }
        }
    } else {
        if (Age < 77.000000000) {
            if (Age < 60.000000000) {
                if (Age < 58.000000000) {
                    return -0.148923665;
                } else {
                    return 1.500845790;
                }
            } else {
                if (Age < 67.000000000) {
                    return -1.106297140;
                } else {
                    return -0.288696945;
                }
            }
        } else {
            if (BMI < 27.472526600) {
                return 1.245852230;
            } else {
                if (Age < 79.000000000) {
                    return -0.455143332;
                } else {
                    return 0.630643249;
                }
            }
        }
    }
}

inline double predict_tree_142(double Gender, double Age, double BMI) {
    if (BMI < 21.952478400) {
        if (BMI < 21.913805000) {
            if (BMI < 21.877550100) {
                if (BMI < 21.545091600) {
                    return 0.018543135;
                } else {
                    return 0.176767409;
                }
            } else {
                return -0.192635462;
            }
        } else {
            return 0.508143127;
        }
    } else {
        if (BMI < 22.100290300) {
            if (BMI < 22.060354200) {
                if (BMI < 21.989893000) {
                    return -0.356857330;
                } else {
                    return -0.148823217;
                }
            } else {
                return -0.448100448;
            }
        } else {
            if (BMI < 22.582708400) {
                if (BMI < 22.546575500) {
                    return 0.052318897;
                } else {
                    return 0.551523685;
                }
            } else {
                if (BMI < 22.640865300) {
                    return -0.454702914;
                } else {
                    return 0.005498466;
                }
            }
        }
    }
}

inline double predict_tree_143(double Gender, double Age, double BMI) {
    if (BMI < 21.096191400) {
        if (Age < 34.000000000) {
            if (Age < 33.000000000) {
                if (Age < 32.000000000) {
                    return 0.046827964;
                } else {
                    return 1.539903880;
                }
            } else {
                return -1.052158590;
            }
        } else {
            if (Age < 50.000000000) {
                if (BMI < 20.504934300) {
                    return 0.014459940;
                } else {
                    return 0.865380526;
                }
            } else {
                if (Age < 61.000000000) {
                    return -0.581619263;
                } else {
                    return 0.874060750;
                }
            }
        }
    } else {
        if (Age < 49.000000000) {
            if (Age < 48.000000000) {
                if (BMI < 25.469387100) {
                    return -0.031734861;
                } else {
                    return 0.065841660;
                }
            } else {
                if (BMI < 26.869806300) {
                    return -0.410467476;
                } else {
                    return 0.879741669;
                }
            }
        } else {
            if (BMI < 24.302486400) {
                if (BMI < 24.251277900) {
                    return 0.101889797;
                } else {
                    return 0.983914137;
                }
            } else {
                if (BMI < 24.677021000) {
                    return -0.185045466;
                } else {
                    return -0.004709168;
                }
            }
        }
    }
}

inline double predict_tree_144(double Gender, double Age, double BMI) {
    if (BMI < 26.312810900) {
        if (BMI < 26.264944100) {
            if (BMI < 26.234567600) {
                if (BMI < 25.306932400) {
                    return -0.005021055;
                } else {
                    return -0.088959724;
                }
            } else {
                return 0.526478350;
            }
        } else {
            if (BMI < 26.303619400) {
                if (BMI < 26.296566000) {
                    return -0.406344622;
                } else {
                    return 0.181456625;
                }
            } else {
                return -0.511131704;
            }
        }
    } else {
        if (BMI < 26.315052000) {
            if (Gender < 1.000000000) {
                return 1.006452440;
            } else {
                return -0.256237477;
            }
        } else {
            if (BMI < 27.450605400) {
                if (BMI < 26.528511000) {
                    return -0.085695125;
                } else {
                    return 0.083625481;
                }
            } else {
                if (BMI < 27.776929900) {
                    return -0.146841690;
                } else {
                    return 0.027929017;
                }
            }
        }
    }
}

inline double predict_tree_145(double Gender, double Age, double BMI) {
    if (BMI < 25.992439300) {
        if (BMI < 25.977846100) {
            if (Age < 31.000000000) {
                if (BMI < 22.837369900) {
                    return 0.093058184;
                } else {
                    return -0.049942680;
                }
            } else {
                if (BMI < 22.100290300) {
                    return -0.057770532;
                } else {
                    return 0.045845013;
                }
            }
        } else {
            if (Age < 69.000000000) {
                if (Age < 45.000000000) {
                    return 1.465700390;
                } else {
                    return 0.551217854;
                }
            } else {
                return -1.265599850;
            }
        }
    } else {
        if (BMI < 26.023048400) {
            if (Age < 60.000000000) {
                if (Age < 46.000000000) {
                    return -0.463156849;
                } else {
                    return -1.827849270;
                }
            } else {
                if (Age < 68.000000000) {
                    return 1.662447930;
                } else {
                    return -0.438354820;
                }
            }
        } else {
            if (Age < 31.000000000) {
                if (Age < 25.000000000) {
                    return -0.145010769;
                } else {
                    return 0.415558726;
                }
            } else {
                if (BMI < 26.030820800) {
                    return 0.509047747;
                } else {
                    return -0.073108241;
                }
            }
        }
    }
}

inline double predict_tree_146(double Gender, double Age, double BMI) {
    if (Age < 78.000000000) {
        if (Age < 72.000000000) {
            if (Age < 69.000000000) {
                if (Age < 64.000000000) {
                    return 0.011327630;
                } else {
                    return 0.067917116;
                }
            } else {
                if (Gender < 1.000000000) {
                    return -0.173410296;
                } else {
                    return -0.069494560;
                }
            }
        } else {
            if (Gender < 1.000000000) {
                if (Age < 77.000000000) {
                    return 0.097115479;
                } else {
                    return 0.356533498;
                }
            } else {
                if (Age < 75.000000000) {
                    return 0.124315351;
                } else {
                    return -0.081440955;
                }
            }
        }
    } else {
        if (Age < 79.000000000) {
            return -0.216179937;
        } else {
            if (Gender < 1.000000000) {
                return -0.136284590;
            } else {
                return 0.249459431;
            }
        }
    }
}

inline double predict_tree_147(double Gender, double Age, double BMI) {
    if (Age < 49.000000000) {
        if (Age < 48.000000000) {
            if (Age < 47.000000000) {
                if (BMI < 26.555965400) {
                    return -0.008659611;
                } else {
                    return -0.194431096;
                }
            } else {
                if (BMI < 23.082542400) {
                    return -0.422632933;
                } else {
                    return 0.382064879;
                }
            }
        } else {
            if (BMI < 26.869806300) {
                if (BMI < 20.957170500) {
                    return 0.834785819;
                } else {
                    return -0.479146689;
                }
            } else {
                if (BMI < 27.084159900) {
                    return 1.455048680;
                } else {
                    return -0.268314660;
                }
            }
        }
    } else {
        if (Age < 51.000000000) {
            if (BMI < 26.234567600) {
                if (BMI < 25.951557200) {
                    return 0.202608988;
                } else {
                    return -0.837706864;
                }
            } else {
                if (BMI < 26.595745100) {
                    return 1.494394180;
                } else {
                    return -0.139556587;
                }
            }
        } else {
            if (BMI < 23.634033200) {
                if (BMI < 23.306680700) {
                    return 0.025109565;
                } else {
                    return 0.328089565;
                }
            } else {
                if (BMI < 23.849777200) {
                    return -0.281371325;
                } else {
                    return 0.020757874;
                }
            }
        }
    }
}

inline double predict_tree_148(double Gender, double Age, double BMI) {
    if (Age < 64.000000000) {
        if (Age < 23.000000000) {
            if (Age < 21.000000000) {
                if (Gender < 1.000000000) {
                    return -0.160196915;
                } else {
                    return 0.031667057;
                }
            } else {
                if (Age < 22.000000000) {
                    return 0.064612858;
                } else {
                    return 0.133354798;
                }
            }
        } else {
            if (Age < 59.000000000) {
                if (Age < 57.000000000) {
                    return -0.013363418;
                } else {
                    return 0.120658889;
                }
            } else {
                if (Age < 61.000000000) {
                    return -0.168409199;
                } else {
                    return -0.007775789;
                }
            }
        }
    } else {
        if (Age < 65.000000000) {
            if (Gender < 1.000000000) {
                return 0.334701926;
            } else {
                return 0.187521741;
            }
        } else {
            if (Age < 79.000000000) {
                if (Age < 78.000000000) {
                    return 0.043531116;
                } else {
                    return -0.158598751;
                }
            } else {
                if (Gender < 1.000000000) {
                    return 0.283273309;
                } else {
                    return 0.064035073;
                }
            }
        }
    }
}

inline double predict_tree_149(double Gender, double Age, double BMI) {
    if (BMI < 27.084159900) {
        if (BMI < 26.851852400) {
            if (BMI < 26.595745100) {
                if (BMI < 26.528511000) {
                    return 0.010600626;
                } else {
                    return 0.158975556;
                }
            } else {
                if (Age < 61.000000000) {
                    return -0.542406976;
                } else {
                    return 0.356150925;
                }
            }
        } else {
            if (Age < 34.000000000) {
                if (BMI < 26.880950900) {
                    return -1.504189610;
                } else {
                    return 0.083591059;
                }
            } else {
                if (Age < 74.000000000) {
                    return 0.446474791;
                } else {
                    return -0.627130926;
                }
            }
        }
    } else {
        if (Age < 77.000000000) {
            if (BMI < 27.137414900) {
                if (Age < 76.000000000) {
                    return -0.824661493;
                } else {
                    return 0.630694509;
                }
            } else {
                if (BMI < 27.450605400) {
                    return -0.023259999;
                } else {
                    return -0.319275618;
                }
            }
        } else {
            if (BMI < 27.137414900) {
                if (Age < 78.000000000) {
                    return -0.714029610;
                } else {
                    return 0.262065411;
                }
            } else {
                if (BMI < 27.472526600) {
                    return 2.114223240;
                } else {
                    return 0.311048090;
                }
            }
        }
    }
}

inline double predict_tree_150(double Gender, double Age, double BMI) {
    if (BMI < 27.776929900) {
        if (BMI < 27.450605400) {
            if (BMI < 27.173914000) {
                if (BMI < 26.282564200) {
                    return -0.002878864;
                } else {
                    return -0.068651386;
                }
            } else {
                if (BMI < 27.379665400) {
                    return 0.404592365;
                } else {
                    return 0.000594091;
                }
            }
        } else {
            if (BMI < 27.472526600) {
                return -0.284594983;
            } else {
                if (BMI < 27.714790300) {
                    return -0.063846730;
                } else {
                    return -0.226210818;
                }
            }
        }
    } else {
        return 0.356953800;
    }
}

inline double predict_tree_151(double Gender, double Age, double BMI) {
    if (BMI < 26.528511000) {
        if (BMI < 26.423570600) {
            if (BMI < 26.312810900) {
                if (BMI < 26.303619400) {
                    return -0.007661102;
                } else {
                    return -0.462014735;
                }
            } else {
                if (BMI < 26.315052000) {
                    return 0.643606305;
                } else {
                    return 0.032856181;
                }
            }
        } else {
            if (Gender < 1.000000000) {
                return -0.369533032;
            } else {
                if (BMI < 26.485540400) {
                    return 0.364392459;
                } else {
                    return -0.449848622;
                }
            }
        }
    } else {
        if (BMI < 26.595745100) {
            if (BMI < 26.592670400) {
                if (Gender < 1.000000000) {
                    return 0.159840316;
                } else {
                    return 0.794989586;
                }
            } else {
                if (Gender < 1.000000000) {
                    return 0.433447599;
                } else {
                    return -0.309599310;
                }
            }
        } else {
            if (BMI < 26.851852400) {
                if (Gender < 1.000000000) {
                    return -0.239273205;
                } else {
                    return 0.637621880;
                }
            } else {
                if (BMI < 27.450605400) {
                    return 0.115052477;
                } else {
                    return -0.075266339;
                }
            }
        }
    }
}

inline double predict_tree_152(double Gender, double Age, double BMI) {
    if (Age < 27.000000000) {
        if (Age < 21.000000000) {
            if (Gender < 1.000000000) {
                return -0.086583123;
            } else {
                return -0.046784982;
            }
        } else {
            if (Age < 24.000000000) {
                if (Gender < 1.000000000) {
                    return 0.043042760;
                } else {
                    return -0.022394797;
                }
            } else {
                if (Age < 26.000000000) {
                    return -0.050394714;
                } else {
                    return -0.011234102;
                }
            }
        }
    } else {
        if (Age < 30.000000000) {
            if (Gender < 1.000000000) {
                if (Age < 29.000000000) {
                    return 0.022057742;
                } else {
                    return -0.012496008;
                }
            } else {
                if (Age < 29.000000000) {
                    return 0.043084793;
                } else {
                    return 0.184107661;
                }
            }
        } else {
            if (Age < 37.000000000) {
                if (Age < 36.000000000) {
                    return -0.007670216;
                } else {
                    return -0.102767929;
                }
            } else {
                if (Age < 40.000000000) {
                    return 0.103297673;
                } else {
                    return 0.010935338;
                }
            }
        }
    }
}

inline double predict_tree_153(double Gender, double Age, double BMI) {
    if (BMI < 24.382715200) {
        if (BMI < 24.355421100) {
            if (BMI < 24.304616900) {
                if (BMI < 24.121749900) {
                    return -0.003025997;
                } else {
                    return 0.176835686;
                }
            } else {
                if (BMI < 24.341758700) {
                    return -0.370573759;
                } else {
                    return 0.169112757;
                }
            }
        } else {
            return 0.646923959;
        }
    } else {
        if (BMI < 24.989587800) {
            if (BMI < 24.976087600) {
                if (BMI < 24.456064200) {
                    return -0.231562346;
                } else {
                    return -0.054237526;
                }
            } else {
                return -0.521401584;
            }
        } else {
            if (BMI < 25.057359700) {
                if (BMI < 24.997549100) {
                    return 0.667606711;
                } else {
                    return 0.293889910;
                }
            } else {
                if (Gender < 1.000000000) {
                    return 0.009016081;
                } else {
                    return -0.180377588;
                }
            }
        }
    }
}

inline double predict_tree_154(double Gender, double Age, double BMI) {
    if (Age < 57.000000000) {
        if (Age < 56.000000000) {
            if (Age < 49.000000000) {
                if (Age < 48.000000000) {
                    return -0.003630144;
                } else {
                    return -0.325995624;
                }
            } else {
                if (Age < 50.000000000) {
                    return 0.186856702;
                } else {
                    return 0.033682868;
                }
            }
        } else {
            if (Gender < 1.000000000) {
                return -0.292441487;
            } else {
                return -0.408562958;
            }
        }
    } else {
        if (Age < 62.000000000) {
            if (Age < 61.000000000) {
                if (Age < 60.000000000) {
                    return 0.135640576;
                } else {
                    return -0.180233479;
                }
            } else {
                if (Gender < 1.000000000) {
                    return 0.498679966;
                } else {
                    return 0.111499354;
                }
            }
        } else {
            if (Age < 64.000000000) {
                if (Gender < 1.000000000) {
                    return -0.216976225;
                } else {
                    return 0.068670824;
                }
            } else {
                if (Age < 65.000000000) {
                    return 0.246942878;
                } else {
                    return 0.009772954;
                }
            }
        }
    }
}

inline double predict_tree_155(double Gender, double Age, double BMI) {
    if (BMI < 26.595745100) {
        if (BMI < 26.528511000) {
            if (BMI < 26.485540400) {
                if (BMI < 21.877550100) {
                    return 0.076843917;
                } else {
                    return -0.009198125;
                }
            } else {
                return -0.574823499;
            }
        } else {
            if (BMI < 26.555965400) {
                return 0.433767647;
            } else {
                if (Gender < 1.000000000) {
                    return 0.133308202;
                } else {
                    return 0.389669985;
                }
            }
        }
    } else {
        if (BMI < 27.173914000) {
            if (BMI < 27.160493900) {
                if (BMI < 27.137414900) {
                    return -0.218637809;
                } else {
                    return 0.152333975;
                }
            } else {
                return -0.384815782;
            }
        } else {
            if (BMI < 27.379665400) {
                return 0.354177803;
            } else {
                if (BMI < 27.714790300) {
                    return -0.181170866;
                } else {
                    return 0.037247043;
                }
            }
        }
    }
}

inline double predict_tree_156(double Gender, double Age, double BMI) {
    if (BMI < 21.218317000) {
        if (BMI < 20.796730000) {
            if (BMI < 20.504934300) {
                return 0.196427837;
            } else {
                return -0.235062569;
            }
        } else {
            if (BMI < 21.096191400) {
                return 0.315114141;
            } else {
                return 0.107370108;
            }
        }
    } else {
        if (BMI < 26.014568300) {
            if (BMI < 25.767219500) {
                if (BMI < 25.762678100) {
                    return -0.008355606;
                } else {
                    return -0.482307017;
                }
            } else {
                if (BMI < 25.795917500) {
                    return 0.439265728;
                } else {
                    return 0.052867655;
                }
            }
        } else {
            if (BMI < 26.023048400) {
                return -1.108898640;
            } else {
                if (BMI < 26.851852400) {
                    return -0.067031227;
                } else {
                    return 0.037062932;
                }
            }
        }
    }
}

inline double predict_tree_157(double Gender, double Age, double BMI) {
    if (BMI < 25.306932400) {
        if (BMI < 25.207756000) {
            if (BMI < 25.057359700) {
                if (BMI < 24.989587800) {
                    return 0.008012286;
                } else {
                    return 0.469171435;
                }
            } else {
                if (BMI < 25.082862900) {
                    return -0.281867772;
                } else {
                    return -0.098830409;
                }
            }
        } else {
            if (Gender < 1.000000000) {
                if (BMI < 25.216253300) {
                    return 1.476604220;
                } else {
                    return 0.164054826;
                }
            } else {
                if (BMI < 25.216253300) {
                    return -1.077619670;
                } else {
                    return 0.185000867;
                }
            }
        }
    } else {
        if (BMI < 25.333334000) {
            return -0.643947661;
        } else {
            if (Gender < 1.000000000) {
                if (BMI < 25.535446200) {
                    return -0.092768244;
                } else {
                    return 0.009158696;
                }
            } else {
                if (BMI < 25.469387100) {
                    return -0.607773602;
                } else {
                    return -0.065552056;
                }
            }
        }
    }
}

inline double predict_tree_158(double Gender, double Age, double BMI) {
    if (BMI < 27.776929900) {
        if (BMI < 23.597003900) {
            if (Gender < 1.000000000) {
                if (BMI < 22.790329000) {
                    return -0.708115876;
                } else {
                    return -0.171609610;
                }
            } else {
                if (BMI < 23.533042900) {
                    return 0.014012389;
                } else {
                    return -0.323771626;
                }
            }
        } else {
            if (BMI < 23.634033200) {
                if (BMI < 23.620288800) {
                    return 0.688709795;
                } else {
                    return 0.442278266;
                }
            } else {
                if (BMI < 23.795360600) {
                    return 0.172250077;
                } else {
                    return 0.011588868;
                }
            }
        }
    } else {
        return -0.339957237;
    }
}

inline double predict_tree_159(double Gender, double Age, double BMI) {
    if (Age < 74.000000000) {
        if (Age < 64.000000000) {
            if (BMI < 27.173914000) {
                if (BMI < 26.880950900) {
                    return -0.008795431;
                } else {
                    return 0.329214096;
                }
            } else {
                if (Age < 53.000000000) {
                    return 0.163204327;
                } else {
                    return -0.696356058;
                }
            }
        } else {
            if (Age < 65.000000000) {
                if (BMI < 23.922422400) {
                    return -0.181742564;
                } else {
                    return 0.629583180;
                }
            } else {
                if (BMI < 23.507804900) {
                    return 0.272789448;
                } else {
                    return -0.039172299;
                }
            }
        }
    } else {
        if (BMI < 25.469387100) {
            if (BMI < 23.833004000) {
                if (Age < 75.000000000) {
                    return 0.599380851;
                } else {
                    return -0.383859098;
                }
            } else {
                if (BMI < 24.323228800) {
                    return 1.326280360;
                } else {
                    return 0.039533563;
                }
            }
        } else {
            if (BMI < 27.084159900) {
                if (BMI < 26.264944100) {
                    return -0.229392678;
                } else {
                    return -0.950658619;
                }
            } else {
                if (Age < 77.000000000) {
                    return -0.083628751;
                } else {
                    return 0.713523626;
                }
            }
        }
    }
}

inline double predict_tree_160(double Gender, double Age, double BMI) {
    if (Age < 63.000000000) {
        if (BMI < 27.173914000) {
            if (BMI < 25.767219500) {
                if (BMI < 25.057359700) {
                    return -0.002800750;
                } else {
                    return -0.084081240;
                }
            } else {
                if (BMI < 25.795917500) {
                    return 0.654611886;
                } else {
                    return 0.057822008;
                }
            }
        } else {
            if (Age < 53.000000000) {
                if (Age < 47.000000000) {
                    return -0.410938859;
                } else {
                    return 0.631294310;
                }
            } else {
                if (BMI < 27.450605400) {
                    return -1.062868830;
                } else {
                    return -0.390557855;
                }
            }
        }
    } else {
        if (Age < 65.000000000) {
            if (BMI < 23.225431400) {
                if (BMI < 23.147254900) {
                    return -0.332776338;
                } else {
                    return -0.980881572;
                }
            } else {
                if (Age < 64.000000000) {
                    return 0.160117730;
                } else {
                    return 0.729197919;
                }
            }
        } else {
            if (BMI < 23.507804900) {
                if (Age < 75.000000000) {
                    return 0.434870392;
                } else {
                    return -0.287650466;
                }
            } else {
                if (BMI < 24.056934400) {
                    return -0.284110844;
                } else {
                    return 0.027536470;
                }
            }
        }
    }
}

inline double predict_tree_161(double Gender, double Age, double BMI) {
    if (Age < 49.000000000) {
        if (Age < 48.000000000) {
            if (Age < 47.000000000) {
                if (Age < 46.000000000) {
                    return -0.025551949;
                } else {
                    return -0.180091858;
                }
            } else {
                if (Gender < 1.000000000) {
                    return 0.376188695;
                } else {
                    return -0.150203556;
                }
            }
        } else {
            return -0.313181818;
        }
    } else {
        if (Gender < 1.000000000) {
            if (Age < 74.000000000) {
                if (Age < 72.000000000) {
                    return 0.091067322;
                } else {
                    return 0.327158362;
                }
            } else {
                if (Age < 77.000000000) {
                    return -0.217746392;
                } else {
                    return 0.088208921;
                }
            }
        } else {
            if (Age < 75.000000000) {
                if (Age < 73.000000000) {
                    return -0.015169423;
                } else {
                    return 0.266924113;
                }
            } else {
                if (Age < 78.000000000) {
                    return -0.220174417;
                } else {
                    return -0.061895955;
                }
            }
        }
    }
}

inline double predict_tree_162(double Gender, double Age, double BMI) {
    if (BMI < 27.776929900) {
        if (Age < 51.000000000) {
            if (BMI < 25.535446200) {
                if (BMI < 25.057359700) {
                    return 0.003353128;
                } else {
                    return -0.121238239;
                }
            } else {
                if (BMI < 26.595745100) {
                    return 0.130317390;
                } else {
                    return -0.150480419;
                }
            }
        } else {
            if (Age < 61.000000000) {
                if (BMI < 26.332889600) {
                    return -0.027083555;
                } else {
                    return -0.346168727;
                }
            } else {
                if (BMI < 27.173914000) {
                    return -0.017044954;
                } else {
                    return 0.441189617;
                }
            }
        }
    } else {
        if (Age < 40.000000000) {
            if (Age < 29.000000000) {
                if (Age < 25.000000000) {
                    return -0.225310817;
                } else {
                    return 0.565183222;
                }
            } else {
                return -1.049313310;
            }
        } else {
            if (Age < 53.000000000) {
                return 1.516320940;
            } else {
                if (Age < 57.000000000) {
                    return -1.395102500;
                } else {
                    return 0.349107385;
                }
            }
        }
    }
}

inline double predict_tree_163(double Gender, double Age, double BMI) {
    if (BMI < 24.989587800) {
        if (BMI < 24.304616900) {
            if (BMI < 24.221452700) {
                if (Age < 71.000000000) {
                    return -0.001839837;
                } else {
                    return 0.195700660;
                }
            } else {
                if (Age < 26.000000000) {
                    return -0.394536018;
                } else {
                    return 0.458728164;
                }
            }
        } else {
            if (Age < 39.000000000) {
                if (Age < 31.000000000) {
                    return -0.165536404;
                } else {
                    return 0.376883924;
                }
            } else {
                if (Age < 62.000000000) {
                    return -0.254381776;
                } else {
                    return -0.001794004;
                }
            }
        }
    } else {
        if (BMI < 24.997549100) {
            if (Age < 69.000000000) {
                if (Age < 32.000000000) {
                    return 0.536392450;
                } else {
                    return 1.372849580;
                }
            } else {
                return -1.266365770;
            }
        } else {
            if (Age < 79.000000000) {
                if (Age < 59.000000000) {
                    return 0.049526848;
                } else {
                    return -0.051118769;
                }
            } else {
                if (BMI < 25.762678100) {
                    return 1.316088200;
                } else {
                    return 0.105789706;
                }
            }
        }
    }
}

inline double predict_tree_164(double Gender, double Age, double BMI) {
    if (BMI < 20.796730000) {
        return -0.204367295;
    } else {
        if (BMI < 21.218317000) {
            if (BMI < 20.957170500) {
                return 0.139734447;
            } else {
                if (BMI < 21.096191400) {
                    return 0.295716196;
                } else {
                    return 0.173447609;
                }
            }
        } else {
            if (BMI < 21.545091600) {
                if (BMI < 21.367521300) {
                    return -0.057696655;
                } else {
                    return -0.215444535;
                }
            } else {
                if (BMI < 21.952478400) {
                    return 0.115636088;
                } else {
                    return 0.008319114;
                }
            }
        }
    }
}

inline double predict_tree_165(double Gender, double Age, double BMI) {
    if (Age < 64.000000000) {
        if (Age < 60.000000000) {
            if (Age < 57.000000000) {
                if (Age < 48.000000000) {
                    return -0.008789699;
                } else {
                    return -0.052648857;
                }
            } else {
                if (Gender < 1.000000000) {
                    return 0.046222266;
                } else {
                    return 0.132925138;
                }
            }
        } else {
            if (Age < 61.000000000) {
                if (Gender < 1.000000000) {
                    return 0.002951309;
                } else {
                    return -0.458620131;
                }
            } else {
                if (Age < 63.000000000) {
                    return 0.013719376;
                } else {
                    return -0.132912368;
                }
            }
        }
    } else {
        if (Gender < 1.000000000) {
            if (Age < 66.000000000) {
                if (Age < 65.000000000) {
                    return 0.227272943;
                } else {
                    return 0.561382413;
                }
            } else {
                if (Age < 67.000000000) {
                    return -0.552904785;
                } else {
                    return 0.080082208;
                }
            }
        } else {
            if (Age < 70.000000000) {
                if (Age < 69.000000000) {
                    return 0.007377570;
                } else {
                    return 0.265800953;
                }
            } else {
                if (Age < 73.000000000) {
                    return -0.305815220;
                } else {
                    return 0.032871377;
                }
            }
        }
    }
}

inline double predict_tree_166(double Gender, double Age, double BMI) {
    if (BMI < 25.306932400) {
        if (Age < 49.000000000) {
            if (Age < 38.000000000) {
                if (Age < 31.000000000) {
                    return -0.012483393;
                } else {
                    return 0.074242681;
                }
            } else {
                if (BMI < 24.560325600) {
                    return -0.025569625;
                } else {
                    return -0.257113457;
                }
            }
        } else {
            if (BMI < 24.956596400) {
                if (BMI < 24.898143800) {
                    return 0.061869647;
                } else {
                    return -0.578805506;
                }
            } else {
                if (Age < 74.000000000) {
                    return 0.191695526;
                } else {
                    return 0.736675262;
                }
            }
        }
    } else {
        if (Age < 50.000000000) {
            if (BMI < 25.469387100) {
                if (Age < 38.000000000) {
                    return -0.635001123;
                } else {
                    return 0.385193944;
                }
            } else {
                if (BMI < 25.992439300) {
                    return 0.165935740;
                } else {
                    return -0.020329114;
                }
            }
        } else {
            if (Age < 54.000000000) {
                if (BMI < 27.776929900) {
                    return -0.538972557;
                } else {
                    return 1.137618660;
                }
            } else {
                if (BMI < 25.381467800) {
                    return -0.719434977;
                } else {
                    return -0.073029704;
                }
            }
        }
    }
}

inline double predict_tree_167(double Gender, double Age, double BMI) {
    if (BMI < 27.450605400) {
        if (BMI < 23.423557300) {
            if (BMI < 23.147254900) {
                if (BMI < 23.082542400) {
                    return -0.035692532;
                } else {
                    return 0.485205382;
                }
            } else {
                if (BMI < 23.225431400) {
                    return -0.492858618;
                } else {
                    return -0.087474890;
                }
            }
        } else {
            if (BMI < 23.795360600) {
                if (BMI < 23.597003900) {
                    return 0.068982378;
                } else {
                    return 0.215125352;
                }
            } else {
                if (BMI < 23.849777200) {
                    return -0.340506613;
                } else {
                    return 0.005521425;
                }
            }
        }
    } else {
        if (BMI < 27.714790300) {
            return -0.575796664;
        } else {
            if (BMI < 27.776929900) {
                return -0.002036080;
            } else {
                return -0.072778240;
            }
        }
    }
}

inline double predict_tree_168(double Gender, double Age, double BMI) {
    if (Gender < 1.000000000) {
        if (Age < 47.000000000) {
            if (Age < 45.000000000) {
                if (Age < 43.000000000) {
                    return 0.000623051;
                } else {
                    return 0.173320860;
                }
            } else {
                if (Age < 46.000000000) {
                    return -0.205851361;
                } else {
                    return -0.072651796;
                }
            }
        } else {
            if (Age < 48.000000000) {
                return 0.449638933;
            } else {
                if (Age < 55.000000000) {
                    return 0.091664903;
                } else {
                    return 0.007008967;
                }
            }
        }
    } else {
        if (Age < 70.000000000) {
            if (Age < 42.000000000) {
                if (Age < 41.000000000) {
                    return -0.009060055;
                } else {
                    return 0.333201051;
                }
            } else {
                if (Age < 44.000000000) {
                    return -0.356833488;
                } else {
                    return 0.004696857;
                }
            }
        } else {
            if (Age < 71.000000000) {
                return -0.448594123;
            } else {
                if (Age < 76.000000000) {
                    return -0.026406877;
                } else {
                    return -0.206423357;
                }
            }
        }
    }
}

inline double predict_tree_169(double Gender, double Age, double BMI) {
    if (Age < 72.000000000) {
        if (Age < 71.000000000) {
            if (Age < 63.000000000) {
                if (Age < 60.000000000) {
                    return -0.001080534;
                } else {
                    return -0.093512587;
                }
            } else {
                if (Age < 65.000000000) {
                    return 0.164161295;
                } else {
                    return -0.019616071;
                }
            }
        } else {
            return -0.236313850;
        }
    } else {
        if (Age < 73.000000000) {
            if (Gender < 1.000000000) {
                return 0.370219618;
            } else {
                return -0.020663571;
            }
        } else {
            if (Gender < 1.000000000) {
                if (Age < 75.000000000) {
                    return -0.316449583;
                } else {
                    return 0.101571828;
                }
            } else {
                if (Age < 76.000000000) {
                    return 0.279997051;
                } else {
                    return -0.099731073;
                }
            }
        }
    }
}

inline double predict_tree_170(double Gender, double Age, double BMI) {
    if (Age < 76.000000000) {
        if (Age < 57.000000000) {
            if (Age < 52.000000000) {
                if (Age < 49.000000000) {
                    return 0.006389570;
                } else {
                    return 0.096618548;
                }
            } else {
                if (Age < 53.000000000) {
                    return -0.227971241;
                } else {
                    return -0.040892575;
                }
            }
        } else {
            if (Gender < 1.000000000) {
                if (Age < 66.000000000) {
                    return 0.137547299;
                } else {
                    return 0.009298437;
                }
            } else {
                if (Age < 73.000000000) {
                    return -0.010789799;
                } else {
                    return 0.136892378;
                }
            }
        }
    } else {
        if (Gender < 1.000000000) {
            if (Age < 78.000000000) {
                if (Age < 77.000000000) {
                    return -0.233976409;
                } else {
                    return -0.072364964;
                }
            } else {
                return -0.254171431;
            }
        } else {
            if (Age < 78.000000000) {
                return -0.194836035;
            } else {
                if (Age < 79.000000000) {
                    return -0.070183136;
                } else {
                    return -0.036021076;
                }
            }
        }
    }
}

inline double predict_tree_171(double Gender, double Age, double BMI) {
    if (BMI < 25.102392200) {
        if (BMI < 24.967113500) {
            if (BMI < 24.898143800) {
                if (BMI < 24.772096600) {
                    return -0.001572203;
                } else {
                    return 0.192935601;
                }
            } else {
                if (Gender < 1.000000000) {
                    return -0.414272308;
                } else {
                    return 0.086876631;
                }
            }
        } else {
            if (BMI < 24.976087600) {
                if (Gender < 1.000000000) {
                    return 0.611171186;
                } else {
                    return 0.199126720;
                }
            } else {
                if (BMI < 24.989587800) {
                    return -0.279111922;
                } else {
                    return 0.273779750;
                }
            }
        }
    } else {
        if (BMI < 25.207756000) {
            if (Gender < 1.000000000) {
                if (BMI < 25.165145900) {
                    return -0.239809111;
                } else {
                    return -0.458295673;
                }
            } else {
                if (BMI < 25.148605300) {
                    return -0.321287364;
                } else {
                    return 0.280775368;
                }
            }
        } else {
            if (BMI < 25.216253300) {
                if (Gender < 1.000000000) {
                    return 1.259286520;
                } else {
                    return -1.042229770;
                }
            } else {
                if (BMI < 27.379665400) {
                    return -0.031265855;
                } else {
                    return -0.202709705;
                }
            }
        }
    }
}

inline double predict_tree_172(double Gender, double Age, double BMI) {
    if (Age < 24.000000000) {
        if (Gender < 1.000000000) {
            if (Age < 23.000000000) {
                if (Age < 22.000000000) {
                    return -0.018671466;
                } else {
                    return 0.004047615;
                }
            } else {
                return 0.065828487;
            }
        } else {
            if (Age < 23.000000000) {
                if (Age < 22.000000000) {
                    return 0.037025578;
                } else {
                    return 0.129432291;
                }
            } else {
                return -0.035349019;
            }
        }
    } else {
        if (Age < 27.000000000) {
            if (Gender < 1.000000000) {
                if (Age < 25.000000000) {
                    return -0.120476484;
                } else {
                    return -0.062727310;
                }
            } else {
                if (Age < 25.000000000) {
                    return 0.043235905;
                } else {
                    return -0.083854467;
                }
            }
        } else {
            if (Gender < 1.000000000) {
                if (Age < 49.000000000) {
                    return -0.011093129;
                } else {
                    return 0.047837004;
                }
            } else {
                if (Age < 60.000000000) {
                    return 0.000628934;
                } else {
                    return -0.068805471;
                }
            }
        }
    }
}

inline double predict_tree_173(double Gender, double Age, double BMI) {
    if (BMI < 25.306932400) {
        if (BMI < 25.227865200) {
            if (BMI < 21.877550100) {
                if (BMI < 21.545091600) {
                    return -0.004410961;
                } else {
                    return 0.219050571;
                }
            } else {
                if (BMI < 22.100290300) {
                    return -0.276662111;
                } else {
                    return 0.018667772;
                }
            }
        } else {
            if (BMI < 25.249895100) {
                if (Gender < 1.000000000) {
                    return 0.169475287;
                } else {
                    return -0.156590417;
                }
            } else {
                if (BMI < 25.280109400) {
                    return 0.294615805;
                } else {
                    return 0.196520016;
                }
            }
        }
    } else {
        if (BMI < 25.333334000) {
            return -0.649408996;
        } else {
            if (Gender < 1.000000000) {
                if (BMI < 25.401701000) {
                    return 0.212343171;
                } else {
                    return -0.009305118;
                }
            } else {
                if (BMI < 26.023048400) {
                    return -0.275801420;
                } else {
                    return 0.207990199;
                }
            }
        }
    }
}

inline double predict_tree_174(double Gender, double Age, double BMI) {
    if (BMI < 24.967113500) {
        if (BMI < 24.569315000) {
            if (BMI < 24.456064200) {
                if (BMI < 24.441803000) {
                    return -0.009321157;
                } else {
                    return -0.372592390;
                }
            } else {
                if (BMI < 24.464601500) {
                    return 0.689803660;
                } else {
                    return 0.134469882;
                }
            }
        } else {
            if (BMI < 24.618103000) {
                if (BMI < 24.609375000) {
                    return -0.242820218;
                } else {
                    return -0.917348564;
                }
            } else {
                if (BMI < 24.677021000) {
                    return 0.188921764;
                } else {
                    return -0.124224059;
                }
            }
        }
    } else {
        if (BMI < 25.057359700) {
            if (BMI < 24.989587800) {
                if (BMI < 24.976087600) {
                    return 0.282225192;
                } else {
                    return -0.261510134;
                }
            } else {
                if (BMI < 24.997549100) {
                    return 0.963549614;
                } else {
                    return 0.293497324;
                }
            }
        } else {
            if (Gender < 1.000000000) {
                if (BMI < 25.207756000) {
                    return -0.154111505;
                } else {
                    return 0.052090354;
                }
            } else {
                if (BMI < 26.794937100) {
                    return -0.126787484;
                } else {
                    return 1.085007670;
                }
            }
        }
    }
}

inline double predict_tree_175(double Gender, double Age, double BMI) {
    if (Age < 78.000000000) {
        if (Age < 73.000000000) {
            if (Age < 69.000000000) {
                if (Age < 47.000000000) {
                    return -0.001099606;
                } else {
                    return 0.029395580;
                }
            } else {
                if (Age < 72.000000000) {
                    return -0.118041478;
                } else {
                    return 0.022579895;
                }
            }
        } else {
            if (Age < 74.000000000) {
                if (Gender < 1.000000000) {
                    return 0.144825473;
                } else {
                    return 0.206866235;
                }
            } else {
                if (Age < 75.000000000) {
                    return -0.080172777;
                } else {
                    return 0.097869426;
                }
            }
        }
    } else {
        if (Age < 79.000000000) {
            if (Gender < 1.000000000) {
                return -0.381672621;
            } else {
                return -0.070223123;
            }
        } else {
            if (Gender < 1.000000000) {
                return -0.077099577;
            } else {
                return 0.086602464;
            }
        }
    }
}

inline double predict_tree_176(double Gender, double Age, double BMI) {
    if (Age < 22.000000000) {
        if (Age < 21.000000000) {
            if (Gender < 1.000000000) {
                return -0.120127685;
            } else {
                return -0.079524644;
            }
        } else {
            if (Gender < 1.000000000) {
                return 0.002872171;
            } else {
                return -0.103001662;
            }
        }
    } else {
        if (Age < 77.000000000) {
            if (Age < 69.000000000) {
                if (Age < 61.000000000) {
                    return -0.007948982;
                } else {
                    return 0.059930716;
                }
            } else {
                if (Age < 72.000000000) {
                    return -0.213226900;
                } else {
                    return -0.014997261;
                }
            }
        } else {
            if (Gender < 1.000000000) {
                if (Age < 78.000000000) {
                    return 0.324488640;
                } else {
                    return 0.110000655;
                }
            } else {
                if (Age < 78.000000000) {
                    return 0.090270773;
                } else {
                    return -0.079334989;
                }
            }
        }
    }
}

inline double predict_tree_177(double Gender, double Age, double BMI) {
    if (Age < 28.000000000) {
        if (Age < 24.000000000) {
            if (Age < 21.000000000) {
                if (Gender < 1.000000000) {
                    return -0.130596697;
                } else {
                    return -0.007708677;
                }
            } else {
                if (Gender < 1.000000000) {
                    return -0.003943311;
                } else {
                    return 0.039681494;
                }
            }
        } else {
            if (Age < 26.000000000) {
                if (Gender < 1.000000000) {
                    return -0.042035971;
                } else {
                    return -0.099767923;
                }
            } else {
                if (Gender < 1.000000000) {
                    return -0.084446766;
                } else {
                    return 0.008171405;
                }
            }
        }
    } else {
        if (Age < 60.000000000) {
            if (Age < 57.000000000) {
                if (Age < 52.000000000) {
                    return 0.021355862;
                } else {
                    return -0.044356208;
                }
            } else {
                if (Age < 58.000000000) {
                    return 0.292929232;
                } else {
                    return 0.042644992;
                }
            }
        } else {
            if (Age < 63.000000000) {
                if (Gender < 1.000000000) {
                    return -0.067917086;
                } else {
                    return -0.226745173;
                }
            } else {
                if (Age < 65.000000000) {
                    return 0.087943658;
                } else {
                    return -0.012841959;
                }
            }
        }
    }
}

inline double predict_tree_178(double Gender, double Age, double BMI) {
    if (BMI < 21.952478400) {
        if (BMI < 21.913805000) {
            if (Age < 22.000000000) {
                if (BMI < 21.877550100) {
                    return -0.247395054;
                } else {
                    return 0.882458091;
                }
            } else {
                if (Age < 28.000000000) {
                    return 0.263596386;
                } else {
                    return 0.002833190;
                }
            }
        } else {
            if (Age < 44.000000000) {
                if (Age < 43.000000000) {
                    return 0.443936557;
                } else {
                    return -1.115597370;
                }
            } else {
                return 1.390980840;
            }
        }
    } else {
        if (BMI < 22.100290300) {
            if (Age < 24.000000000) {
                if (BMI < 21.989893000) {
                    return 1.170846100;
                } else {
                    return -0.284571081;
                }
            } else {
                if (Age < 54.000000000) {
                    return -0.546851814;
                } else {
                    return 0.111641958;
                }
            }
        } else {
            if (BMI < 22.189348200) {
                if (Age < 22.000000000) {
                    return -1.665489910;
                } else {
                    return 0.600136995;
                }
            } else {
                if (BMI < 22.230987500) {
                    return -0.460000247;
                } else {
                    return -0.022405602;
                }
            }
        }
    }
}

inline double predict_tree_179(double Gender, double Age, double BMI) {
    if (Age < 74.000000000) {
        if (Age < 35.000000000) {
            if (Age < 34.000000000) {
                if (Age < 27.000000000) {
                    return -0.025488665;
                } else {
                    return -0.003612573;
                }
            } else {
                if (Gender < 1.000000000) {
                    return -0.004721494;
                } else {
                    return -0.161798432;
                }
            }
        } else {
            if (Age < 69.000000000) {
                if (Age < 40.000000000) {
                    return 0.067486837;
                } else {
                    return 0.029712584;
                }
            } else {
                if (Gender < 1.000000000) {
                    return 0.020818304;
                } else {
                    return -0.151180595;
                }
            }
        }
    } else {
        if (Age < 75.000000000) {
            if (Gender < 1.000000000) {
                return -0.839742243;
            } else {
                return 0.051797375;
            }
        } else {
            if (Age < 76.000000000) {
                if (Gender < 1.000000000) {
                    return 0.388180405;
                } else {
                    return -0.184577778;
                }
            } else {
                if (Age < 79.000000000) {
                    return -0.198971212;
                } else {
                    return -0.009031641;
                }
            }
        }
    }
}

inline double predict_tree_180(double Gender, double Age, double BMI) {
    if (BMI < 26.595745100) {
        if (BMI < 26.528511000) {
            if (BMI < 24.989587800) {
                if (BMI < 24.976087600) {
                    return -0.006555466;
                } else {
                    return -0.585799277;
                }
            } else {
                if (BMI < 25.057359700) {
                    return 0.536157012;
                } else {
                    return 0.013753346;
                }
            }
        } else {
            if (Age < 65.000000000) {
                if (Age < 49.000000000) {
                    return 0.124529205;
                } else {
                    return 0.898209989;
                }
            } else {
                if (BMI < 26.570304900) {
                    return -0.875934660;
                } else {
                    return 0.239252701;
                }
            }
        }
    } else {
        if (Age < 72.000000000) {
            if (Age < 71.000000000) {
                if (Age < 68.000000000) {
                    return -0.180810958;
                } else {
                    return 0.334322780;
                }
            } else {
                if (BMI < 26.880950900) {
                    return 0.714647949;
                } else {
                    return -1.669786450;
                }
            }
        } else {
            if (Age < 74.000000000) {
                if (BMI < 26.979530300) {
                    return 2.301610470;
                } else {
                    return -0.026176643;
                }
            } else {
                if (BMI < 27.084159900) {
                    return -0.526392579;
                } else {
                    return 0.361379683;
                }
            }
        }
    }
}

inline double predict_tree_181(double Gender, double Age, double BMI) {
    if (BMI < 24.271844900) {
        if (Age < 70.000000000) {
            if (BMI < 24.221452700) {
                if (BMI < 23.893259000) {
                    return 0.027935481;
                } else {
                    return -0.093745217;
                }
            } else {
                if (Age < 23.000000000) {
                    return -0.603142917;
                } else {
                    return 0.451258123;
                }
            }
        } else {
            if (Age < 78.000000000) {
                if (BMI < 22.758306500) {
                    return -0.148101389;
                } else {
                    return 0.528677464;
                }
            } else {
                if (BMI < 23.051754000) {
                    return 0.058071006;
                } else {
                    return -0.667267144;
                }
            }
        }
    } else {
        if (BMI < 24.670375800) {
            if (Age < 53.000000000) {
                if (Age < 50.000000000) {
                    return -0.039710037;
                } else {
                    return 0.821631312;
                }
            } else {
                if (Age < 75.000000000) {
                    return -0.544212461;
                } else {
                    return 0.674388051;
                }
            }
        } else {
            if (Age < 31.000000000) {
                if (Age < 27.000000000) {
                    return -0.055085681;
                } else {
                    return 0.285031080;
                }
            } else {
                if (Age < 43.000000000) {
                    return -0.113282546;
                } else {
                    return 0.000172668;
                }
            }
        }
    }
}

inline double predict_tree_182(double Gender, double Age, double BMI) {
    if (BMI < 21.877550100) {
        if (Age < 22.000000000) {
            if (BMI < 21.829952200) {
                if (BMI < 21.773841900) {
                    return -0.224722087;
                } else {
                    return 0.210828543;
                }
            } else {
                return -1.069280270;
            }
        } else {
            if (Age < 28.000000000) {
                if (BMI < 20.957170500) {
                    return -0.083776355;
                } else {
                    return 0.478301674;
                }
            } else {
                if (Age < 53.000000000) {
                    return -0.031889312;
                } else {
                    return 0.530805111;
                }
            }
        }
    } else {
        if (BMI < 21.913805000) {
            if (Age < 25.000000000) {
                return 0.832538128;
            } else {
                if (Age < 32.000000000) {
                    return -1.315510270;
                } else {
                    return -0.522693336;
                }
            }
        } else {
            if (BMI < 21.952478400) {
                if (Age < 50.000000000) {
                    return 0.271084011;
                } else {
                    return 1.448259000;
                }
            } else {
                if (BMI < 22.100290300) {
                    return -0.306822360;
                } else {
                    return -0.000132925;
                }
            }
        }
    }
}

inline double predict_tree_183(double Gender, double Age, double BMI) {
    if (BMI < 23.795360600) {
        if (BMI < 23.711845400) {
            if (Gender < 1.000000000) {
                if (BMI < 23.634033200) {
                    return -0.152932465;
                } else {
                    return -0.887138605;
                }
            } else {
                if (BMI < 23.620288800) {
                    return 0.047347616;
                } else {
                    return 0.336671591;
                }
            }
        } else {
            if (Gender < 1.000000000) {
                if (BMI < 23.722810700) {
                    return 0.280451983;
                } else {
                    return 0.903559685;
                }
            } else {
                if (BMI < 23.733238200) {
                    return 0.389687598;
                } else {
                    return -0.074565321;
                }
            }
        }
    } else {
        if (BMI < 23.849777200) {
            if (BMI < 23.808797800) {
                return -0.237543792;
            } else {
                if (Gender < 1.000000000) {
                    return -0.157688171;
                } else {
                    return -0.862653077;
                }
            }
        } else {
            if (BMI < 23.893259000) {
                return 0.503920555;
            } else {
                if (BMI < 23.939481700) {
                    return -0.341311753;
                } else {
                    return -0.016830996;
                }
            }
        }
    }
}

inline double predict_tree_184(double Gender, double Age, double BMI) {
    if (Age < 47.000000000) {
        if (Age < 38.000000000) {
            if (Age < 28.000000000) {
                if (Age < 23.000000000) {
                    return -0.000937530;
                } else {
                    return -0.043299206;
                }
            } else {
                if (Age < 30.000000000) {
                    return 0.054688077;
                } else {
                    return 0.014660378;
                }
            }
        } else {
            if (Age < 39.000000000) {
                if (Gender < 1.000000000) {
                    return -0.048545994;
                } else {
                    return -0.212989420;
                }
            } else {
                if (Age < 40.000000000) {
                    return 0.072162300;
                } else {
                    return -0.032000676;
                }
            }
        }
    } else {
        if (Age < 50.000000000) {
            if (Age < 49.000000000) {
                if (Age < 48.000000000) {
                    return 0.169725984;
                } else {
                    return -0.052303642;
                }
            } else {
                if (Gender < 1.000000000) {
                    return 0.292471915;
                } else {
                    return 0.080582842;
                }
            }
        } else {
            if (Age < 54.000000000) {
                if (Age < 52.000000000) {
                    return 0.007796607;
                } else {
                    return -0.084730290;
                }
            } else {
                if (Age < 55.000000000) {
                    return 0.247879326;
                } else {
                    return 0.014007785;
                }
            }
        }
    }
}

inline double predict_tree_185(double Gender, double Age, double BMI) {
    if (BMI < 26.827421200) {
        if (BMI < 22.282478300) {
            if (BMI < 22.189348200) {
                if (BMI < 22.100290300) {
                    return -0.033571511;
                } else {
                    return 0.224306479;
                }
            } else {
                if (Gender < 1.000000000) {
                    return 0.834796131;
                } else {
                    return -0.398416370;
                }
            }
        } else {
            if (BMI < 22.491350200) {
                if (Gender < 1.000000000) {
                    return -0.599047184;
                } else {
                    return 0.217028216;
                }
            } else {
                if (BMI < 22.521507300) {
                    return -0.346488446;
                } else {
                    return -0.000801973;
                }
            }
        }
    } else {
        if (BMI < 26.851852400) {
            return -0.565246522;
        } else {
            if (BMI < 27.137414900) {
                if (BMI < 26.869806300) {
                    return 0.104328677;
                } else {
                    return -0.242111683;
                }
            } else {
                if (BMI < 27.379665400) {
                    return 0.153237924;
                } else {
                    return -0.079926595;
                }
            }
        }
    }
}

inline double predict_tree_186(double Gender, double Age, double BMI) {
    if (BMI < 20.504934300) {
        return 0.284810990;
    } else {
        if (BMI < 20.796730000) {
            return -0.231266931;
        } else {
            if (BMI < 22.582708400) {
                if (BMI < 22.230987500) {
                    return -0.016923230;
                } else {
                    return 0.192700386;
                }
            } else {
                if (BMI < 22.640865300) {
                    return -0.333876193;
                } else {
                    return 0.003375960;
                }
            }
        }
    }
}

inline double predict_tree_187(double Gender, double Age, double BMI) {
    if (Age < 73.000000000) {
        if (Age < 69.000000000) {
            if (Age < 64.000000000) {
                if (Age < 62.000000000) {
                    return -0.007159373;
                } else {
                    return -0.102198906;
                }
            } else {
                if (Age < 67.000000000) {
                    return 0.072190277;
                } else {
                    return 0.175388917;
                }
            }
        } else {
            if (Gender < 1.000000000) {
                if (Age < 72.000000000) {
                    return -0.192127734;
                } else {
                    return 0.225780651;
                }
            } else {
                if (Age < 70.000000000) {
                    return -0.040535435;
                } else {
                    return -0.346003801;
                }
            }
        }
    } else {
        if (Age < 74.000000000) {
            if (Gender < 1.000000000) {
                return 0.330071032;
            } else {
                return 0.186958581;
            }
        } else {
            if (Age < 79.000000000) {
                if (Age < 78.000000000) {
                    return 0.043269705;
                } else {
                    return -0.073085718;
                }
            } else {
                if (Gender < 1.000000000) {
                    return 0.149289623;
                } else {
                    return 0.343496084;
                }
            }
        }
    }
}

inline double predict_tree_188(double Gender, double Age, double BMI) {
    if (Age < 73.000000000) {
        if (Age < 71.000000000) {
            if (Age < 64.000000000) {
                if (Age < 58.000000000) {
                    return 0.005312881;
                } else {
                    return -0.092493169;
                }
            } else {
                if (Age < 65.000000000) {
                    return 0.251941293;
                } else {
                    return 0.020794259;
                }
            }
        } else {
            if (Age < 72.000000000) {
                return -0.222942799;
            } else {
                if (Gender < 1.000000000) {
                    return -0.035680648;
                } else {
                    return -0.167412609;
                }
            }
        }
    } else {
        if (Age < 79.000000000) {
            if (Age < 76.000000000) {
                if (Age < 75.000000000) {
                    return 0.069844089;
                } else {
                    return 0.134629786;
                }
            } else {
                if (Age < 77.000000000) {
                    return -0.075261593;
                } else {
                    return 0.048283596;
                }
            }
        } else {
            return 0.131270841;
        }
    }
}

inline double predict_tree_189(double Gender, double Age, double BMI) {
    if (BMI < 25.992439300) {
        if (BMI < 23.597003900) {
            if (Age < 66.000000000) {
                if (BMI < 23.335466400) {
                    return 0.010303668;
                } else {
                    return -0.207276881;
                }
            } else {
                if (BMI < 22.598140700) {
                    return -0.197758704;
                } else {
                    return 0.305927426;
                }
            }
        } else {
            if (BMI < 23.634033200) {
                if (Age < 70.000000000) {
                    return 0.600996137;
                } else {
                    return -0.699452221;
                }
            } else {
                if (Age < 28.000000000) {
                    return -0.054647386;
                } else {
                    return 0.066474207;
                }
            }
        }
    } else {
        if (BMI < 26.023048400) {
            if (Age < 33.000000000) {
                if (Age < 22.000000000) {
                    return -1.252123590;
                } else {
                    return 0.519212544;
                }
            } else {
                if (Age < 60.000000000) {
                    return -1.241614100;
                } else {
                    return 0.037355427;
                }
            }
        } else {
            if (Age < 62.000000000) {
                if (Age < 61.000000000) {
                    return -0.016835961;
                } else {
                    return 0.852629066;
                }
            } else {
                if (BMI < 27.160493900) {
                    return -0.234026566;
                } else {
                    return 0.112200305;
                }
            }
        }
    }
}

inline double predict_tree_190(double Gender, double Age, double BMI) {
    if (BMI < 27.173914000) {
        if (BMI < 21.952478400) {
            if (BMI < 21.913805000) {
                if (BMI < 21.877550100) {
                    return 0.079477049;
                } else {
                    return -0.379992336;
                }
            } else {
                return 0.574337363;
            }
        } else {
            if (BMI < 22.100290300) {
                if (Gender < 1.000000000) {
                    return -0.594224989;
                } else {
                    return -0.265046000;
                }
            } else {
                if (BMI < 22.145328500) {
                    return 0.288103580;
                } else {
                    return 0.000669995;
                }
            }
        }
    } else {
        if (BMI < 27.450605400) {
            if (BMI < 27.379665400) {
                return 0.412162781;
            } else {
                return 0.215127155;
            }
        } else {
            if (BMI < 27.776929900) {
                if (BMI < 27.472526600) {
                    return -0.021175392;
                } else {
                    return 0.025792263;
                }
            } else {
                return 0.264629602;
            }
        }
    }
}

inline double predict_tree_191(double Gender, double Age, double BMI) {
    if (Age < 79.000000000) {
        if (Age < 31.000000000) {
            if (Gender < 1.000000000) {
                if (Age < 29.000000000) {
                    return -0.035703979;
                } else {
                    return -0.218657702;
                }
            } else {
                if (Age < 23.000000000) {
                    return 0.070893533;
                } else {
                    return 0.000057206;
                }
            }
        } else {
            if (Age < 36.000000000) {
                if (Age < 35.000000000) {
                    return 0.042122398;
                } else {
                    return 0.141102090;
                }
            } else {
                if (Age < 41.000000000) {
                    return -0.055366434;
                } else {
                    return 0.007073862;
                }
            }
        }
    } else {
        if (Gender < 1.000000000) {
            return 0.350685000;
        } else {
            return 0.037195116;
        }
    }
}

inline double predict_tree_192(double Gender, double Age, double BMI) {
    if (BMI < 23.423557300) {
        if (Gender < 1.000000000) {
            if (BMI < 23.082542400) {
                if (BMI < 23.046875000) {
                    return -0.302228451;
                } else {
                    return 1.570361610;
                }
            } else {
                if (BMI < 23.388687100) {
                    return -0.678724170;
                } else {
                    return -0.069027662;
                }
            }
        } else {
            if (BMI < 23.374725300) {
                if (BMI < 22.230987500) {
                    return -0.050173610;
                } else {
                    return 0.022310821;
                }
            } else {
                if (BMI < 23.413110700) {
                    return -0.453376979;
                } else {
                    return -0.195151567;
                }
            }
        }
    } else {
        if (BMI < 23.795360600) {
            if (BMI < 23.711845400) {
                if (Gender < 1.000000000) {
                    return -0.175145656;
                } else {
                    return 0.171883881;
                }
            } else {
                if (Gender < 1.000000000) {
                    return 0.619085431;
                } else {
                    return 0.130054846;
                }
            }
        } else {
            if (BMI < 23.833004000) {
                if (BMI < 23.808691000) {
                    return -0.356144965;
                } else {
                    return -0.512038171;
                }
            } else {
                if (BMI < 23.922422400) {
                    return 0.246803030;
                } else {
                    return -0.004360662;
                }
            }
        }
    }
}

inline double predict_tree_193(double Gender, double Age, double BMI) {
    if (BMI < 25.755157500) {
        if (BMI < 24.560325600) {
            if (BMI < 24.456064200) {
                if (BMI < 24.441803000) {
                    return 0.003058383;
                } else {
                    return -0.347793311;
                }
            } else {
                if (BMI < 24.464601500) {
                    return 0.597212136;
                } else {
                    return 0.155799299;
                }
            }
        } else {
            if (BMI < 24.618103000) {
                if (BMI < 24.609375000) {
                    return -0.258305132;
                } else {
                    return -1.000947120;
                }
            } else {
                if (BMI < 25.057359700) {
                    return 0.006077799;
                } else {
                    return -0.073787928;
                }
            }
        }
    } else {
        if (BMI < 25.762678100) {
            return 0.648340762;
        } else {
            if (BMI < 25.992439300) {
                if (BMI < 25.880571400) {
                    return -0.033698812;
                } else {
                    return 0.340491086;
                }
            } else {
                if (BMI < 26.023048400) {
                    return -0.452865541;
                } else {
                    return 0.013658715;
                }
            }
        }
    }
}

inline double predict_tree_194(double Gender, double Age, double BMI) {
    if (Age < 48.000000000) {
        if (Age < 37.000000000) {
            if (Age < 33.000000000) {
                if (Age < 26.000000000) {
                    return -0.005932773;
                } else {
                    return 0.024205754;
                }
            } else {
                if (Age < 35.000000000) {
                    return -0.057223987;
                } else {
                    return -0.003370289;
                }
            }
        } else {
            if (Age < 42.000000000) {
                if (Age < 41.000000000) {
                    return 0.022266038;
                } else {
                    return 0.194373965;
                }
            } else {
                if (Age < 43.000000000) {
                    return -0.167472333;
                } else {
                    return 0.041569062;
                }
            }
        }
    } else {
        if (Age < 49.000000000) {
            if (Gender < 1.000000000) {
                return -0.300058186;
            } else {
                return -0.136787280;
            }
        } else {
            if (Gender < 1.000000000) {
                if (Age < 50.000000000) {
                    return 0.253772944;
                } else {
                    return 0.015436737;
                }
            } else {
                if (Age < 57.000000000) {
                    return -0.119492196;
                } else {
                    return -0.025352165;
                }
            }
        }
    }
}

inline double predict_tree_195(double Gender, double Age, double BMI) {
    if (BMI < 27.450605400) {
        if (BMI < 27.173914000) {
            if (BMI < 27.160493900) {
                if (BMI < 23.437500000) {
                    return -0.028517161;
                } else {
                    return 0.012855287;
                }
            } else {
                return -0.396490723;
            }
        } else {
            if (BMI < 27.379665400) {
                return 0.402993381;
            } else {
                return 0.135168910;
            }
        }
    } else {
        if (BMI < 27.472526600) {
            return -0.386297762;
        } else {
            if (BMI < 27.714790300) {
                return -0.165994868;
            } else {
                if (BMI < 27.776929900) {
                    return 0.007329677;
                } else {
                    return -0.037029922;
                }
            }
        }
    }
}

inline double predict_tree_196(double Gender, double Age, double BMI) {
    if (BMI < 20.504934300) {
        if (Age < 21.000000000) {
            return 0.970970511;
        } else {
            if (Age < 22.000000000) {
                return -0.860447824;
            } else {
                if (Age < 26.000000000) {
                    return 0.639646947;
                } else {
                    return -0.067818366;
                }
            }
        }
    } else {
        if (BMI < 25.767219500) {
            if (Age < 76.000000000) {
                if (BMI < 25.306932400) {
                    return 0.005353355;
                } else {
                    return -0.103863761;
                }
            } else {
                if (BMI < 23.456790900) {
                    return -0.219581619;
                } else {
                    return 0.379386574;
                }
            }
        } else {
            if (Age < 76.000000000) {
                if (BMI < 25.992439300) {
                    return 0.317980975;
                } else {
                    return 0.017931201;
                }
            } else {
                if (BMI < 26.555965400) {
                    return -0.693606853;
                } else {
                    return 0.111501083;
                }
            }
        }
    }
}

inline double predict_tree_197(double Gender, double Age, double BMI) {
    if (Age < 39.000000000) {
        if (Gender < 1.000000000) {
            if (Age < 26.000000000) {
                if (Age < 23.000000000) {
                    return -0.051471367;
                } else {
                    return 0.096742265;
                }
            } else {
                if (Age < 31.000000000) {
                    return -0.132465973;
                } else {
                    return -0.014776827;
                }
            }
        } else {
            if (Age < 36.000000000) {
                if (Age < 35.000000000) {
                    return 0.008046478;
                } else {
                    return 0.303309977;
                }
            } else {
                if (Age < 37.000000000) {
                    return -0.267921239;
                } else {
                    return 0.039518855;
                }
            }
        }
    } else {
        if (Age < 42.000000000) {
            if (Gender < 1.000000000) {
                if (Age < 40.000000000) {
                    return 0.097315192;
                } else {
                    return 0.031843875;
                }
            } else {
                if (Age < 41.000000000) {
                    return 0.112693518;
                } else {
                    return 0.273999810;
                }
            }
        } else {
            if (Age < 73.000000000) {
                if (Age < 69.000000000) {
                    return 0.029969074;
                } else {
                    return -0.083104409;
                }
            } else {
                if (Age < 78.000000000) {
                    return 0.084707685;
                } else {
                    return 0.003573832;
                }
            }
        }
    }
}

inline double predict_tree_198(double Gender, double Age, double BMI) {
    if (Gender < 1.000000000) {
        if (Age < 39.000000000) {
            if (Age < 37.000000000) {
                if (Age < 36.000000000) {
                    return -0.010563319;
                } else {
                    return 0.157173768;
                }
            } else {
                return -0.159108222;
            }
        } else {
            if (Age < 55.000000000) {
                if (Age < 54.000000000) {
                    return 0.076401323;
                } else {
                    return 0.576931059;
                }
            } else {
                if (Age < 75.000000000) {
                    return -0.031606145;
                } else {
                    return 0.153052151;
                }
            }
        }
    } else {
        if (Age < 31.000000000) {
            if (Age < 29.000000000) {
                if (Age < 23.000000000) {
                    return 0.049689572;
                } else {
                    return -0.001620737;
                }
            } else {
                return 0.166109011;
            }
        } else {
            if (Age < 70.000000000) {
                if (Age < 66.000000000) {
                    return -0.054256719;
                } else {
                    return 0.072174959;
                }
            } else {
                if (Age < 71.000000000) {
                    return -0.472843915;
                } else {
                    return -0.088099584;
                }
            }
        }
    }
}

inline double predict_tree_199(double Gender, double Age, double BMI) {
    if (BMI < 23.795360600) {
        if (BMI < 23.711845400) {
            if (Gender < 1.000000000) {
                if (BMI < 23.634033200) {
                    return -0.113006882;
                } else {
                    return -0.875009954;
                }
            } else {
                if (BMI < 23.597003900) {
                    return 0.022037366;
                } else {
                    return 0.314422071;
                }
            }
        } else {
            if (Gender < 1.000000000) {
                return 0.723873913;
            } else {
                if (BMI < 23.733238200) {
                    return 0.294802010;
                } else {
                    return -0.124066763;
                }
            }
        }
    } else {
        if (BMI < 23.849777200) {
            if (BMI < 23.808797800) {
                if (Gender < 1.000000000) {
                    return -0.518037081;
                } else {
                    return -0.064738676;
                }
            } else {
                if (Gender < 1.000000000) {
                    return 0.022625880;
                } else {
                    return -0.732842863;
                }
            }
        } else {
            if (BMI < 23.893259000) {
                if (BMI < 23.875433000) {
                    return 0.117953502;
                } else {
                    return 0.377735466;
                }
            } else {
                if (BMI < 23.939481700) {
                    return -0.395157456;
                } else {
                    return -0.003961426;
                }
            }
        }
    }
}

inline double predict_tree_200(double Gender, double Age, double BMI) {
    if (Age < 44.000000000) {
        if (Age < 38.000000000) {
            if (Age < 31.000000000) {
                if (Age < 23.000000000) {
                    return 0.016654469;
                } else {
                    return -0.012546224;
                }
            } else {
                if (Age < 36.000000000) {
                    return 0.030453278;
                } else {
                    return -0.017660154;
                }
            }
        } else {
            if (Age < 41.000000000) {
                if (Age < 40.000000000) {
                    return -0.031400424;
                } else {
                    return -0.191450819;
                }
            } else {
                if (Gender < 1.000000000) {
                    return 0.034308456;
                } else {
                    return -0.044304181;
                }
            }
        }
    } else {
        if (Age < 50.000000000) {
            if (Age < 49.000000000) {
                if (Age < 48.000000000) {
                    return 0.099853829;
                } else {
                    return -0.156220645;
                }
            } else {
                if (Gender < 1.000000000) {
                    return 0.437340945;
                } else {
                    return 0.039293125;
                }
            }
        } else {
            if (Age < 54.000000000) {
                if (Age < 52.000000000) {
                    return 0.024244124;
                } else {
                    return -0.158158749;
                }
            } else {
                if (Age < 55.000000000) {
                    return 0.162170783;
                } else {
                    return 0.009062102;
                }
            }
        }
    }
}

inline double predict_tree_201(double Gender, double Age, double BMI) {
    if (Age < 76.000000000) {
        if (Age < 73.000000000) {
            if (Age < 71.000000000) {
                if (Age < 28.000000000) {
                    return -0.018773867;
                } else {
                    return 0.026511127;
                }
            } else {
                if (BMI < 24.456064200) {
                    return 0.249430835;
                } else {
                    return -0.519612491;
                }
            }
        } else {
            if (BMI < 27.379665400) {
                if (BMI < 26.078971900) {
                    return 0.000481305;
                } else {
                    return 0.815900922;
                }
            } else {
                if (BMI < 27.776929900) {
                    return -1.442216280;
                } else {
                    return 0.952170789;
                }
            }
        }
    } else {
        if (BMI < 25.436466200) {
            if (BMI < 22.829963700) {
                if (BMI < 22.189348200) {
                    return 0.234989807;
                } else {
                    return -0.864684939;
                }
            } else {
                if (BMI < 24.323228800) {
                    return 0.564035356;
                } else {
                    return 0.081047639;
                }
            }
        } else {
            if (BMI < 27.084159900) {
                if (Age < 79.000000000) {
                    return -0.928379178;
                } else {
                    return -0.144342586;
                }
            } else {
                if (BMI < 27.472526600) {
                    return 0.973273575;
                } else {
                    return -0.051298201;
                }
            }
        }
    }
}

inline double predict_tree_202(double Gender, double Age, double BMI) {
    if (BMI < 21.218317000) {
        if (Age < 23.000000000) {
            if (Age < 21.000000000) {
                if (BMI < 20.504934300) {
                    return 0.940465152;
                } else {
                    return -0.166976482;
                }
            } else {
                if (BMI < 20.796730000) {
                    return -1.145899650;
                } else {
                    return 0.138276294;
                }
            }
        } else {
            if (Age < 26.000000000) {
                if (BMI < 21.096191400) {
                    return 0.331367552;
                } else {
                    return 0.987490773;
                }
            } else {
                if (Age < 65.000000000) {
                    return 0.063109167;
                } else {
                    return 1.002745390;
                }
            }
        }
    } else {
        if (Age < 23.000000000) {
            if (BMI < 21.367521300) {
                if (BMI < 21.303949400) {
                    return 0.119365841;
                } else {
                    return 1.104422690;
                }
            } else {
                if (BMI < 22.282478300) {
                    return -0.255154192;
                } else {
                    return 0.082799710;
                }
            }
        } else {
            if (Age < 26.000000000) {
                if (BMI < 24.088222500) {
                    return 0.000560194;
                } else {
                    return -0.242985174;
                }
            } else {
                if (BMI < 23.456790900) {
                    return -0.053198602;
                } else {
                    return 0.000755354;
                }
            }
        }
    }
}

inline double predict_tree_203(double Gender, double Age, double BMI) {
    if (BMI < 25.306932400) {
        if (BMI < 25.207756000) {
            if (BMI < 21.218317000) {
                if (BMI < 20.796730000) {
                    return 0.071855798;
                } else {
                    return 0.185585588;
                }
            } else {
                if (BMI < 24.967113500) {
                    return 0.008653106;
                } else {
                    return 0.094323665;
                }
            }
        } else {
            if (Gender < 1.000000000) {
                if (BMI < 25.216253300) {
                    return 1.430666800;
                } else {
                    return 0.233807638;
                }
            } else {
                if (BMI < 25.280109400) {
                    return -0.670025170;
                } else {
                    return 0.719723344;
                }
            }
        }
    } else {
        if (BMI < 25.333334000) {
            return -0.525057137;
        } else {
            if (Gender < 1.000000000) {
                if (BMI < 27.084159900) {
                    return 0.008798333;
                } else {
                    return -0.069066063;
                }
            } else {
                if (BMI < 26.528511000) {
                    return -0.178724036;
                } else {
                    return 1.052928570;
                }
            }
        }
    }
}

inline double predict_tree_204(double Gender, double Age, double BMI) {
    if (BMI < 27.379665400) {
        if (BMI < 27.173914000) {
            if (BMI < 27.160493900) {
                if (BMI < 26.794937100) {
                    return -0.000393313;
                } else {
                    return 0.112778969;
                }
            } else {
                return -0.243368179;
            }
        } else {
            return 0.415961295;
        }
    } else {
        if (BMI < 27.472526600) {
            return -0.084816031;
        } else {
            if (BMI < 27.776929900) {
                return -0.209956795;
            } else {
                return -0.107654370;
            }
        }
    }
}

inline double predict_tree_205(double Gender, double Age, double BMI) {
    if (Age < 38.000000000) {
        if (Gender < 1.000000000) {
            if (Age < 35.000000000) {
                if (Age < 31.000000000) {
                    return -0.015213474;
                } else {
                    return 0.047341779;
                }
            } else {
                if (Age < 36.000000000) {
                    return -0.192380577;
                } else {
                    return -0.021368571;
                }
            }
        } else {
            if (Age < 37.000000000) {
                if (Age < 35.000000000) {
                    return 0.007941654;
                } else {
                    return 0.117662169;
                }
            } else {
                return 0.308755517;
            }
        }
    } else {
        if (Age < 41.000000000) {
            if (Age < 40.000000000) {
                if (Gender < 1.000000000) {
                    return 0.007411406;
                } else {
                    return -0.071689360;
                }
            } else {
                if (Gender < 1.000000000) {
                    return -0.195347801;
                } else {
                    return -0.260636240;
                }
            }
        } else {
            if (Age < 42.000000000) {
                if (Gender < 1.000000000) {
                    return 0.020074829;
                } else {
                    return 0.305664271;
                }
            } else {
                if (Gender < 1.000000000) {
                    return 0.004316319;
                } else {
                    return -0.036558796;
                }
            }
        }
    }
}

inline double predict_tree_206(double Gender, double Age, double BMI) {
    if (BMI < 23.456790900) {
        if (BMI < 23.323417700) {
            if (BMI < 23.306680700) {
                if (BMI < 23.245985000) {
                    return -0.003910185;
                } else {
                    return -0.282626092;
                }
            } else {
                if (Age < 68.000000000) {
                    return 0.652877569;
                } else {
                    return -0.787149787;
                }
            }
        } else {
            if (Age < 62.000000000) {
                if (Age < 21.000000000) {
                    return 0.349442780;
                } else {
                    return -0.403011501;
                }
            } else {
                if (Age < 77.000000000) {
                    return 0.593963325;
                } else {
                    return -1.102176900;
                }
            }
        }
    } else {
        if (BMI < 23.795360600) {
            if (BMI < 23.722810700) {
                if (Age < 25.000000000) {
                    return 0.387763470;
                } else {
                    return -0.020316389;
                }
            } else {
                if (Age < 57.000000000) {
                    return 0.553647995;
                } else {
                    return -0.296841651;
                }
            }
        } else {
            if (BMI < 23.849777200) {
                if (Age < 61.000000000) {
                    return -0.188144922;
                } else {
                    return -1.181389810;
                }
            } else {
                if (BMI < 23.893259000) {
                    return 0.463957816;
                } else {
                    return 0.003515412;
                }
            }
        }
    }
}

inline double predict_tree_207(double Gender, double Age, double BMI) {
    if (BMI < 23.147254900) {
        if (BMI < 23.082542400) {
            if (Age < 75.000000000) {
                if (Age < 71.000000000) {
                    return 0.017257499;
                } else {
                    return 0.920498788;
                }
            } else {
                if (BMI < 22.829963700) {
                    return -0.947847247;
                } else {
                    return -0.158587858;
                }
            }
        } else {
            if (Age < 58.000000000) {
                if (Age < 36.000000000) {
                    return 0.530086994;
                } else {
                    return -0.091404825;
                }
            } else {
                return 1.547638180;
            }
        }
    } else {
        if (BMI < 23.225431400) {
            if (Age < 32.000000000) {
                if (Age < 31.000000000) {
                    return -0.291727453;
                } else {
                    return 0.630316675;
                }
            } else {
                if (Age < 73.000000000) {
                    return -0.883816957;
                } else {
                    return 0.023674859;
                }
            }
        } else {
            if (BMI < 23.243408200) {
                if (Age < 57.000000000) {
                    return 0.630619407;
                } else {
                    return -0.115809180;
                }
            } else {
                if (BMI < 25.280109400) {
                    return 0.004268700;
                } else {
                    return -0.036245558;
                }
            }
        }
    }
}

inline double predict_tree_208(double Gender, double Age, double BMI) {
    if (BMI < 27.379665400) {
        if (BMI < 27.173914000) {
            if (Gender < 1.000000000) {
                if (BMI < 22.790329000) {
                    return -0.770857275;
                } else {
                    return 0.027545437;
                }
            } else {
                if (BMI < 25.029760400) {
                    return -0.002093355;
                } else {
                    return -0.157901213;
                }
            }
        } else {
            return 0.457439274;
        }
    } else {
        if (BMI < 27.472526600) {
            if (BMI < 27.450605400) {
                return -0.135326236;
            } else {
                return -0.267701745;
            }
        } else {
            if (BMI < 27.714790300) {
                return 0.028972754;
            } else {
                return -0.152989045;
            }
        }
    }
}

inline double predict_tree_209(double Gender, double Age, double BMI) {
    if (Age < 25.000000000) {
        if (BMI < 22.313278200) {
            if (Age < 23.000000000) {
                if (BMI < 22.060354200) {
                    return -0.290290624;
                } else {
                    return -0.681775451;
                }
            } else {
                if (BMI < 22.189348200) {
                    return 0.144890249;
                } else {
                    return -0.920650482;
                }
            }
        } else {
            if (BMI < 25.241788900) {
                if (BMI < 24.654832800) {
                    return -0.011323854;
                } else {
                    return 0.309365630;
                }
            } else {
                if (BMI < 25.535446200) {
                    return -0.520992517;
                } else {
                    return -0.074423805;
                }
            }
        }
    } else {
        if (BMI < 21.877550100) {
            if (Age < 26.000000000) {
                if (BMI < 21.367521300) {
                    return 1.019578930;
                } else {
                    return 0.414728463;
                }
            } else {
                if (BMI < 21.545091600) {
                    return -0.086693406;
                } else {
                    return 0.317335844;
                }
            }
        } else {
            if (BMI < 21.913805000) {
                if (Age < 39.000000000) {
                    return -0.243161350;
                } else {
                    return -1.175635810;
                }
            } else {
                if (BMI < 22.100290300) {
                    return -0.185648605;
                } else {
                    return 0.000741154;
                }
            }
        }
    }
}

inline double predict_tree_210(double Gender, double Age, double BMI) {
    if (BMI < 27.450605400) {
        if (BMI < 27.173914000) {
            if (BMI < 23.922422400) {
                if (BMI < 23.849777200) {
                    return 0.009355096;
                } else {
                    return 0.350470275;
                }
            } else {
                if (BMI < 23.939481700) {
                    return -0.537048161;
                } else {
                    return -0.016137589;
                }
            }
        } else {
            return 0.485141754;
        }
    } else {
        if (BMI < 27.472526600) {
            return -0.376018643;
        } else {
            if (BMI < 27.714790300) {
                return -0.155751735;
            } else {
                if (BMI < 27.776929900) {
                    return 0.017969010;
                } else {
                    return -0.128116027;
                }
            }
        }
    }
}

inline double predict_tree_211(double Gender, double Age, double BMI) {
    if (Age < 24.000000000) {
        if (Age < 21.000000000) {
            if (Gender < 1.000000000) {
                return -0.034659434;
            } else {
                return 0.046391983;
            }
        } else {
            if (Gender < 1.000000000) {
                if (Age < 23.000000000) {
                    return 0.019397259;
                } else {
                    return 0.183396339;
                }
            } else {
                if (Age < 23.000000000) {
                    return 0.095298201;
                } else {
                    return -0.102790728;
                }
            }
        }
    } else {
        if (Age < 28.000000000) {
            if (Age < 25.000000000) {
                if (Gender < 1.000000000) {
                    return -0.143685982;
                } else {
                    return -0.009140382;
                }
            } else {
                if (Gender < 1.000000000) {
                    return 0.006957097;
                } else {
                    return -0.056353878;
                }
            }
        } else {
            if (Age < 29.000000000) {
                if (Gender < 1.000000000) {
                    return 0.126711711;
                } else {
                    return 0.182464078;
                }
            } else {
                if (Age < 67.000000000) {
                    return -0.004144916;
                } else {
                    return 0.033809196;
                }
            }
        }
    }
}

inline double predict_tree_212(double Gender, double Age, double BMI) {
    if (Age < 39.000000000) {
        if (Age < 38.000000000) {
            if (Age < 23.000000000) {
                if (Age < 22.000000000) {
                    return -0.021420168;
                } else {
                    return 0.068785220;
                }
            } else {
                if (Age < 28.000000000) {
                    return -0.057982653;
                } else {
                    return -0.007962749;
                }
            }
        } else {
            if (Gender < 1.000000000) {
                return -0.058975406;
            } else {
                return -0.192770749;
            }
        }
    } else {
        if (Age < 42.000000000) {
            if (Age < 41.000000000) {
                if (Age < 40.000000000) {
                    return 0.156193629;
                } else {
                    return -0.096158333;
                }
            } else {
                if (Gender < 1.000000000) {
                    return 0.128753409;
                } else {
                    return 0.320699543;
                }
            }
        } else {
            if (Gender < 1.000000000) {
                if (Age < 44.000000000) {
                    return 0.229966000;
                } else {
                    return 0.026109954;
                }
            } else {
                if (Age < 44.000000000) {
                    return -0.283783644;
                } else {
                    return -0.005078412;
                }
            }
        }
    }
}

inline double predict_tree_213(double Gender, double Age, double BMI) {
    if (BMI < 27.450605400) {
        if (BMI < 22.714681600) {
            if (Age < 75.000000000) {
                if (Age < 71.000000000) {
                    return -0.051149748;
                } else {
                    return 0.767233133;
                }
            } else {
                if (Age < 78.000000000) {
                    return -1.620735880;
                } else {
                    return -0.053969279;
                }
            }
        } else {
            if (BMI < 22.837369900) {
                if (Age < 27.000000000) {
                    return 0.711252272;
                } else {
                    return 0.106107794;
                }
            } else {
                if (BMI < 22.981901200) {
                    return -0.193961427;
                } else {
                    return 0.012064065;
                }
            }
        }
    } else {
        if (Age < 77.000000000) {
            if (Age < 52.000000000) {
                if (Age < 38.000000000) {
                    return -0.485865116;
                } else {
                    return 0.563480258;
                }
            } else {
                if (Age < 71.000000000) {
                    return -0.513722897;
                } else {
                    return -1.282897000;
                }
            }
        } else {
            if (BMI < 27.472526600) {
                return 1.042645340;
            } else {
                if (Age < 78.000000000) {
                    return 1.152548550;
                } else {
                    return -0.410276055;
                }
            }
        }
    }
}

inline double predict_tree_214(double Gender, double Age, double BMI) {
    if (Age < 32.000000000) {
        if (Age < 31.000000000) {
            if (Gender < 1.000000000) {
                if (Age < 27.000000000) {
                    return -0.010652581;
                } else {
                    return -0.106008582;
                }
            } else {
                if (Age < 28.000000000) {
                    return 0.001182247;
                } else {
                    return 0.115997627;
                }
            }
        } else {
            if (Gender < 1.000000000) {
                return -0.083546206;
            } else {
                return -0.168818563;
            }
        }
    } else {
        if (Gender < 1.000000000) {
            if (Age < 61.000000000) {
                if (Age < 60.000000000) {
                    return 0.043702621;
                } else {
                    return -0.258542687;
                }
            } else {
                if (Age < 66.000000000) {
                    return 0.247024745;
                } else {
                    return 0.054217618;
                }
            }
        } else {
            if (Age < 70.000000000) {
                if (Age < 42.000000000) {
                    return 0.051823292;
                } else {
                    return -0.009029180;
                }
            } else {
                if (Age < 73.000000000) {
                    return -0.211611226;
                } else {
                    return -0.047479276;
                }
            }
        }
    }
}

inline double predict_tree_215(double Gender, double Age, double BMI) {
    if (BMI < 22.491350200) {
        if (BMI < 22.230987500) {
            if (Age < 22.000000000) {
                if (BMI < 22.060354200) {
                    return -0.187432840;
                } else {
                    return -0.771530569;
                }
            } else {
                if (BMI < 22.189348200) {
                    return 0.092958488;
                } else {
                    return -0.275386125;
                }
            }
        } else {
            if (Age < 73.000000000) {
                if (Age < 56.000000000) {
                    return 0.164877474;
                } else {
                    return 1.037808300;
                }
            } else {
                if (BMI < 22.405876200) {
                    return -1.177526240;
                } else {
                    return -0.264700621;
                }
            }
        }
    } else {
        if (BMI < 23.225431400) {
            if (BMI < 23.183391600) {
                if (Age < 48.000000000) {
                    return -0.159986570;
                } else {
                    return 0.151911065;
                }
            } else {
                if (Age < 27.000000000) {
                    return -0.122140758;
                } else {
                    return -0.900565803;
                }
            }
        } else {
            if (BMI < 23.323417700) {
                if (BMI < 23.306680700) {
                    return 0.110195674;
                } else {
                    return 0.851880610;
                }
            } else {
                if (BMI < 23.423557300) {
                    return -0.194234654;
                } else {
                    return 0.013074258;
                }
            }
        }
    }
}

inline double predict_tree_216(double Gender, double Age, double BMI) {
    if (BMI < 26.528511000) {
        if (BMI < 26.423570600) {
            if (BMI < 26.312810900) {
                if (BMI < 26.282564200) {
                    return -0.008830852;
                } else {
                    return -0.311436743;
                }
            } else {
                if (BMI < 26.315052000) {
                    return 0.758879781;
                } else {
                    return 0.070928201;
                }
            }
        } else {
            if (Gender < 1.000000000) {
                if (BMI < 26.485540400) {
                    return -0.547574937;
                } else {
                    return -0.267283648;
                }
            } else {
                if (BMI < 26.485540400) {
                    return 0.367709845;
                } else {
                    return -0.419530749;
                }
            }
        }
    } else {
        if (Gender < 1.000000000) {
            if (BMI < 26.570304900) {
                return 0.199300572;
            } else {
                if (BMI < 26.592670400) {
                    return -0.122916609;
                } else {
                    return 0.050349373;
                }
            }
        } else {
            if (BMI < 26.794937100) {
                if (BMI < 26.713068000) {
                    return 0.645101011;
                } else {
                    return -0.795571804;
                }
            } else {
                return 1.056803820;
            }
        }
    }
}

inline double predict_tree_217(double Gender, double Age, double BMI) {
    if (BMI < 27.776929900) {
        if (BMI < 27.450605400) {
            if (BMI < 27.173914000) {
                if (BMI < 27.084159900) {
                    return 0.003664163;
                } else {
                    return -0.184942350;
                }
            } else {
                if (BMI < 27.379665400) {
                    return 0.245397493;
                } else {
                    return 0.159262761;
                }
            }
        } else {
            if (BMI < 27.472526600) {
                return -0.293016970;
            } else {
                if (BMI < 27.714790300) {
                    return -0.042904623;
                } else {
                    return -0.261069030;
                }
            }
        }
    } else {
        return 0.307176083;
    }
}

inline double predict_tree_218(double Gender, double Age, double BMI) {
    if (Age < 75.000000000) {
        if (Age < 71.000000000) {
            if (Age < 68.000000000) {
                if (Age < 56.000000000) {
                    return -0.010161580;
                } else {
                    return -0.072591312;
                }
            } else {
                if (Gender < 1.000000000) {
                    return 0.212263793;
                } else {
                    return -0.004087215;
                }
            }
        } else {
            if (Age < 73.000000000) {
                return -0.264733583;
            } else {
                if (Gender < 1.000000000) {
                    return -0.189364940;
                } else {
                    return 0.247638777;
                }
            }
        }
    } else {
        if (Gender < 1.000000000) {
            if (Age < 78.000000000) {
                if (Age < 76.000000000) {
                    return 0.438622236;
                } else {
                    return 0.240935758;
                }
            } else {
                if (Age < 79.000000000) {
                    return -0.012641347;
                } else {
                    return 0.123353899;
                }
            }
        } else {
            if (Age < 79.000000000) {
                if (Age < 77.000000000) {
                    return 0.016236587;
                } else {
                    return -0.163223818;
                }
            } else {
                return 0.136496171;
            }
        }
    }
}

inline double predict_tree_219(double Gender, double Age, double BMI) {
    if (BMI < 20.504934300) {
        if (Age < 21.000000000) {
            return 1.076069710;
        } else {
            if (Age < 24.000000000) {
                return -0.806107759;
            } else {
                if (Age < 26.000000000) {
                    return 0.996477008;
                } else {
                    return 0.219787762;
                }
            }
        }
    } else {
        if (BMI < 20.796730000) {
            if (Age < 24.000000000) {
                if (Age < 21.000000000) {
                    return -0.082267687;
                } else {
                    return -1.188598510;
                }
            } else {
                if (Age < 25.000000000) {
                    return 1.075603720;
                } else {
                    return -0.462281793;
                }
            }
        } else {
            if (BMI < 21.952478400) {
                if (Age < 71.000000000) {
                    return 0.046554383;
                } else {
                    return 1.123962520;
                }
            } else {
                if (BMI < 22.230987500) {
                    return -0.169077188;
                } else {
                    return -0.019730745;
                }
            }
        }
    }
}

inline double predict_tree_220(double Gender, double Age, double BMI) {
    if (BMI < 22.907121700) {
        if (BMI < 22.790329000) {
            if (Gender < 1.000000000) {
                if (BMI < 22.720438000) {
                    return -0.246685714;
                } else {
                    return -1.493025900;
                }
            } else {
                if (BMI < 22.714681600) {
                    return 0.024023225;
                } else {
                    return 0.184960932;
                }
            }
        } else {
            if (BMI < 22.837369900) {
                if (BMI < 22.829963700) {
                    return 0.517770767;
                } else {
                    return 1.013348580;
                }
            } else {
                if (Gender < 1.000000000) {
                    return -0.153051347;
                } else {
                    return 0.086788699;
                }
            }
        }
    } else {
        if (BMI < 22.948116300) {
            return -0.418679088;
        } else {
            if (BMI < 25.755157500) {
                if (BMI < 25.711662300) {
                    return -0.014371278;
                } else {
                    return -0.272307068;
                }
            } else {
                if (BMI < 25.795917500) {
                    return 0.310975045;
                } else {
                    return 0.025110241;
                }
            }
        }
    }
}

inline double predict_tree_221(double Gender, double Age, double BMI) {
    if (BMI < 20.796730000) {
        if (Age < 21.000000000) {
            if (BMI < 20.504934300) {
                return 0.745266914;
            } else {
                return -0.081685252;
            }
        } else {
            if (Age < 24.000000000) {
                if (Age < 22.000000000) {
                    return -1.347798940;
                } else {
                    return -0.632246315;
                }
            } else {
                if (Age < 26.000000000) {
                    return 0.911798358;
                } else {
                    return -0.302746266;
                }
            }
        }
    } else {
        if (BMI < 21.096191400) {
            if (Age < 21.000000000) {
                return -0.418935448;
            } else {
                if (Age < 51.000000000) {
                    return 0.384639740;
                } else {
                    return -0.200772241;
                }
            }
        } else {
            if (BMI < 22.100290300) {
                if (Age < 53.000000000) {
                    return -0.120978877;
                } else {
                    return 0.197272494;
                }
            } else {
                if (BMI < 22.145328500) {
                    return 0.440529615;
                } else {
                    return -0.007432653;
                }
            }
        }
    }
}

inline double predict_tree_222(double Gender, double Age, double BMI) {
    if (BMI < 24.560325600) {
        if (BMI < 24.515596400) {
            if (Age < 72.000000000) {
                if (Age < 67.000000000) {
                    return 0.000652534;
                } else {
                    return -0.181184307;
                }
            } else {
                if (BMI < 24.056934400) {
                    return 0.028989030;
                } else {
                    return 0.632419646;
                }
            }
        } else {
            if (Age < 45.000000000) {
                if (Age < 31.000000000) {
                    return -0.365245342;
                } else {
                    return 1.667511700;
                }
            } else {
                if (Age < 63.000000000) {
                    return -0.562918782;
                } else {
                    return 0.771390378;
                }
            }
        }
    } else {
        if (BMI < 24.618103000) {
            if (BMI < 24.609375000) {
                if (Age < 39.000000000) {
                    return 0.377905756;
                } else {
                    return -0.581192553;
                }
            } else {
                if (Age < 48.000000000) {
                    return -1.281180380;
                } else {
                    return -0.125450402;
                }
            }
        } else {
            if (Age < 69.000000000) {
                if (Age < 56.000000000) {
                    return -0.036202259;
                } else {
                    return 0.135418355;
                }
            } else {
                if (BMI < 26.592670400) {
                    return -0.195672110;
                } else {
                    return 0.113208562;
                }
            }
        }
    }
}

inline double predict_tree_223(double Gender, double Age, double BMI) {
    if (BMI < 26.595745100) {
        if (BMI < 25.755157500) {
            if (BMI < 25.725517300) {
                if (BMI < 25.711662300) {
                    return 0.006096893;
                } else {
                    return 0.350073159;
                }
            } else {
                return -0.270566672;
            }
        } else {
            if (BMI < 25.762678100) {
                return 0.788502812;
            } else {
                if (BMI < 25.767219500) {
                    return -0.406764299;
                } else {
                    return 0.055078864;
                }
            }
        }
    } else {
        if (BMI < 27.173914000) {
            if (Gender < 1.000000000) {
                if (BMI < 26.869806300) {
                    return -0.063193366;
                } else {
                    return -0.220279485;
                }
            } else {
                if (BMI < 26.794937100) {
                    return 0.131727859;
                } else {
                    return 0.920107841;
                }
            }
        } else {
            if (BMI < 27.450605400) {
                return 0.306757659;
            } else {
                if (BMI < 27.776929900) {
                    return -0.133587569;
                } else {
                    return 0.159563601;
                }
            }
        }
    }
}

inline double predict_tree_224(double Gender, double Age, double BMI) {
    if (Age < 23.000000000) {
        if (BMI < 20.796730000) {
            if (Age < 21.000000000) {
                if (BMI < 20.504934300) {
                    return 0.868210375;
                } else {
                    return -0.080935895;
                }
            } else {
                return -1.111686470;
            }
        } else {
            if (BMI < 22.907121700) {
                if (BMI < 22.675737400) {
                    return 0.122953385;
                } else {
                    return 1.086618420;
                }
            } else {
                if (BMI < 22.991689700) {
                    return -1.293283460;
                } else {
                    return 0.087407231;
                }
            }
        }
    } else {
        if (BMI < 21.218317000) {
            if (Age < 26.000000000) {
                if (BMI < 20.504934300) {
                    return 0.090770125;
                } else {
                    return 1.043324110;
                }
            } else {
                if (Age < 59.000000000) {
                    return -0.035913605;
                } else {
                    return 0.729806662;
                }
            }
        } else {
            if (Age < 57.000000000) {
                if (BMI < 21.469150500) {
                    return -0.288754851;
                } else {
                    return -0.010114690;
                }
            } else {
                if (BMI < 26.880950900) {
                    return 0.063818447;
                } else {
                    return -0.101452641;
                }
            }
        }
    }
}

inline double predict_tree_225(double Gender, double Age, double BMI) {
    if (BMI < 25.755157500) {
        if (BMI < 25.661151900) {
            if (Age < 73.000000000) {
                if (BMI < 25.617284800) {
                    return -0.023814917;
                } else {
                    return 0.282462031;
                }
            } else {
                if (BMI < 22.829963700) {
                    return -0.314732850;
                } else {
                    return 0.148248494;
                }
            }
        } else {
            if (Age < 73.000000000) {
                if (Age < 70.000000000) {
                    return -0.231473580;
                } else {
                    return 1.233162400;
                }
            } else {
                if (Age < 78.000000000) {
                    return -1.880265950;
                } else {
                    return 0.337321937;
                }
            }
        }
    } else {
        if (Age < 76.000000000) {
            if (Age < 75.000000000) {
                if (Age < 74.000000000) {
                    return 0.059920669;
                } else {
                    return -0.485727042;
                }
            } else {
                if (BMI < 26.264944100) {
                    return 2.098495240;
                } else {
                    return 0.078758649;
                }
            }
        } else {
            if (BMI < 27.084159900) {
                if (BMI < 25.762678100) {
                    return 1.460187910;
                } else {
                    return -0.689304471;
                }
            } else {
                if (BMI < 27.472526600) {
                    return 0.923731923;
                } else {
                    return -0.063018776;
                }
            }
        }
    }
}

inline double predict_tree_226(double Gender, double Age, double BMI) {
    if (Age < 78.000000000) {
        if (Age < 24.000000000) {
            if (Gender < 1.000000000) {
                if (Age < 23.000000000) {
                    return 0.015556893;
                } else {
                    return 0.149158195;
                }
            } else {
                if (Age < 23.000000000) {
                    return 0.098431841;
                } else {
                    return -0.074941054;
                }
            }
        } else {
            if (Age < 47.000000000) {
                if (Age < 42.000000000) {
                    return -0.010612851;
                } else {
                    return -0.055013139;
                }
            } else {
                if (Age < 48.000000000) {
                    return 0.213309094;
                } else {
                    return 0.006245439;
                }
            }
        }
    } else {
        if (Age < 79.000000000) {
            if (Gender < 1.000000000) {
                return -0.400858879;
            } else {
                return -0.120531633;
            }
        } else {
            if (Gender < 1.000000000) {
                return 0.068628907;
            } else {
                return -0.189387947;
            }
        }
    }
}

inline double predict_tree_227(double Gender, double Age, double BMI) {
    if (BMI < 22.862533600) {
        if (Age < 75.000000000) {
            if (Age < 72.000000000) {
                if (Age < 69.000000000) {
                    return 0.046480272;
                } else {
                    return -0.757592916;
                }
            } else {
                if (BMI < 22.837369900) {
                    return 0.865654707;
                } else {
                    return -0.640111327;
                }
            }
        } else {
            if (Age < 77.000000000) {
                return -1.440421460;
            } else {
                if (BMI < 22.521507300) {
                    return -0.516395628;
                } else {
                    return 0.590241253;
                }
            }
        }
    } else {
        if (BMI < 22.981901200) {
            if (Age < 48.000000000) {
                if (Age < 23.000000000) {
                    return -1.385241270;
                } else {
                    return -0.372463167;
                }
            } else {
                if (Age < 54.000000000) {
                    return 1.117062210;
                } else {
                    return 0.072292619;
                }
            }
        } else {
            if (BMI < 23.147254900) {
                if (Age < 41.000000000) {
                    return -0.051739078;
                } else {
                    return 0.413777769;
                }
            } else {
                if (BMI < 23.225431400) {
                    return -0.565713823;
                } else {
                    return -0.004065918;
                }
            }
        }
    }
}

inline double predict_tree_228(double Gender, double Age, double BMI) {
    if (BMI < 21.469150500) {
        if (BMI < 21.367521300) {
            if (BMI < 20.957170500) {
                if (BMI < 20.504934300) {
                    return 0.036715787;
                } else {
                    return -0.224675119;
                }
            } else {
                if (BMI < 21.096191400) {
                    return 0.115198903;
                } else {
                    return 0.030487584;
                }
            }
        } else {
            return -0.406195670;
        }
    } else {
        if (BMI < 25.306932400) {
            if (BMI < 25.207756000) {
                if (BMI < 25.057359700) {
                    return 0.008781624;
                } else {
                    return -0.156242982;
                }
            } else {
                if (BMI < 25.216253300) {
                    return 0.601031303;
                } else {
                    return 0.141561851;
                }
            }
        } else {
            if (BMI < 25.333334000) {
                return -0.463683188;
            } else {
                if (BMI < 25.401701000) {
                    return 0.120606899;
                } else {
                    return -0.022329185;
                }
            }
        }
    }
}

inline double predict_tree_229(double Gender, double Age, double BMI) {
    if (BMI < 26.592670400) {
        if (BMI < 26.423570600) {
            if (BMI < 26.312810900) {
                if (BMI < 25.992439300) {
                    return 0.004934307;
                } else {
                    return -0.098689459;
                }
            } else {
                if (BMI < 26.315052000) {
                    return 0.930579960;
                } else {
                    return 0.051216245;
                }
            }
        } else {
            if (Gender < 1.000000000) {
                if (BMI < 26.485540400) {
                    return -0.337127417;
                } else {
                    return -0.135549664;
                }
            } else {
                if (BMI < 26.528511000) {
                    return -0.086569175;
                } else {
                    return 0.776684403;
                }
            }
        }
    } else {
        if (BMI < 26.869806300) {
            if (BMI < 26.851852400) {
                if (BMI < 26.595745100) {
                    return 0.324498236;
                } else {
                    return 0.042031068;
                }
            } else {
                return 0.408013791;
            }
        } else {
            if (BMI < 27.173914000) {
                if (BMI < 27.084159900) {
                    return 0.016271654;
                } else {
                    return -0.145412400;
                }
            } else {
                if (BMI < 27.450605400) {
                    return 0.218599394;
                } else {
                    return 0.009311173;
                }
            }
        }
    }
}

inline double predict_tree_230(double Gender, double Age, double BMI) {
    if (Age < 51.000000000) {
        if (BMI < 26.880950900) {
            if (BMI < 26.869806300) {
                if (BMI < 24.017253900) {
                    return -0.004779386;
                } else {
                    return 0.044537511;
                }
            } else {
                if (Age < 41.000000000) {
                    return -1.290116910;
                } else {
                    return 0.272348613;
                }
            }
        } else {
            if (Age < 39.000000000) {
                if (Age < 29.000000000) {
                    return 0.722700179;
                } else {
                    return -0.454937935;
                }
            } else {
                if (Age < 44.000000000) {
                    return 1.253666760;
                } else {
                    return 0.211279005;
                }
            }
        }
    } else {
        if (Age < 61.000000000) {
            if (BMI < 24.034406700) {
                if (BMI < 23.507804900) {
                    return -0.056702174;
                } else {
                    return 0.333781421;
                }
            } else {
                if (BMI < 24.212293600) {
                    return -0.783145428;
                } else {
                    return -0.160474196;
                }
            }
        } else {
            if (BMI < 22.862533600) {
                if (Age < 65.000000000) {
                    return -0.612530708;
                } else {
                    return 0.034904912;
                }
            } else {
                if (BMI < 23.507804900) {
                    return 0.341995507;
                } else {
                    return -0.008793935;
                }
            }
        }
    }
}

inline double predict_tree_231(double Gender, double Age, double BMI) {
    if (BMI < 27.776929900) {
        if (BMI < 25.306932400) {
            if (BMI < 24.989587800) {
                if (BMI < 24.913494100) {
                    return -0.002364331;
                } else {
                    return -0.236649722;
                }
            } else {
                if (BMI < 24.997549100) {
                    return 0.938926876;
                } else {
                    return 0.074042626;
                }
            }
        } else {
            if (BMI < 25.767219500) {
                if (BMI < 25.762678100) {
                    return -0.106696948;
                } else {
                    return -0.470772773;
                }
            } else {
                if (BMI < 25.795917500) {
                    return 0.541914701;
                } else {
                    return -0.010318669;
                }
            }
        }
    } else {
        return 0.218567550;
    }
}

inline double predict_tree_232(double Gender, double Age, double BMI) {
    if (Age < 77.000000000) {
        if (Age < 49.000000000) {
            if (Age < 48.000000000) {
                if (Age < 45.000000000) {
                    return 0.009349344;
                } else {
                    return 0.059113301;
                }
            } else {
                if (Gender < 1.000000000) {
                    return -0.128608376;
                } else {
                    return -0.176924810;
                }
            }
        } else {
            if (Age < 50.000000000) {
                return 0.301951319;
            } else {
                if (Age < 60.000000000) {
                    return 0.045727115;
                } else {
                    return -0.000714609;
                }
            }
        }
    } else {
        if (Gender < 1.000000000) {
            if (Age < 78.000000000) {
                return 0.409963787;
            } else {
                if (Age < 79.000000000) {
                    return -0.002198382;
                } else {
                    return 0.156174451;
                }
            }
        } else {
            if (Age < 78.000000000) {
                return -0.184364036;
            } else {
                if (Age < 79.000000000) {
                    return 0.093605049;
                } else {
                    return 0.028632183;
                }
            }
        }
    }
}

inline double predict_tree_233(double Gender, double Age, double BMI) {
    if (BMI < 23.795360600) {
        if (BMI < 23.711845400) {
            if (BMI < 23.634033200) {
                if (BMI < 23.597003900) {
                    return -0.009296847;
                } else {
                    return 0.561052799;
                }
            } else {
                if (Age < 21.000000000) {
                    return 0.522362232;
                } else {
                    return -0.360983193;
                }
            }
        } else {
            if (Age < 55.000000000) {
                if (Age < 41.000000000) {
                    return 0.235499084;
                } else {
                    return 1.127274160;
                }
            } else {
                if (Age < 67.000000000) {
                    return -0.746885359;
                } else {
                    return 0.595316887;
                }
            }
        }
    } else {
        if (BMI < 23.849777200) {
            if (Age < 61.000000000) {
                if (Age < 50.000000000) {
                    return -0.361360997;
                } else {
                    return 1.248046040;
                }
            } else {
                if (BMI < 23.833004000) {
                    return -1.238619570;
                } else {
                    return 0.200123057;
                }
            }
        } else {
            if (BMI < 23.922422400) {
                if (Age < 59.000000000) {
                    return 0.157485113;
                } else {
                    return 0.867589474;
                }
            } else {
                if (BMI < 23.939481700) {
                    return -0.417863488;
                } else {
                    return -0.007169473;
                }
            }
        }
    }
}

inline double predict_tree_234(double Gender, double Age, double BMI) {
    if (BMI < 27.714790300) {
        if (BMI < 23.147254900) {
            if (BMI < 23.082542400) {
                if (Gender < 1.000000000) {
                    return -0.178877920;
                } else {
                    return 0.020787686;
                }
            } else {
                if (Gender < 1.000000000) {
                    return -0.174165487;
                } else {
                    return 0.456070632;
                }
            }
        } else {
            if (BMI < 23.225431400) {
                if (BMI < 23.183391600) {
                    return -0.102500826;
                } else {
                    return -0.809254825;
                }
            } else {
                if (BMI < 23.323417700) {
                    return 0.168551683;
                } else {
                    return -0.009603156;
                }
            }
        }
    } else {
        if (BMI < 27.776929900) {
            return -0.148980379;
        } else {
            return -0.246232033;
        }
    }
}

inline double predict_tree_235(double Gender, double Age, double BMI) {
    if (Age < 64.000000000) {
        if (BMI < 26.595745100) {
            if (BMI < 26.528511000) {
                if (BMI < 26.423570600) {
                    return -0.015372575;
                } else {
                    return -0.296118975;
                }
            } else {
                if (Age < 33.000000000) {
                    return 0.732099116;
                } else {
                    return 0.036906138;
                }
            }
        } else {
            if (BMI < 26.880950900) {
                if (Age < 24.000000000) {
                    return -1.207155820;
                } else {
                    return -0.244799137;
                }
            } else {
                if (Age < 29.000000000) {
                    return 0.762332082;
                } else {
                    return -0.163875237;
                }
            }
        }
    } else {
        if (BMI < 27.173914000) {
            if (BMI < 26.880950900) {
                if (BMI < 26.794937100) {
                    return 0.004882501;
                } else {
                    return 0.976344526;
                }
            } else {
                if (Age < 73.000000000) {
                    return -1.152801160;
                } else {
                    return -0.008169648;
                }
            }
        } else {
            if (BMI < 27.450605400) {
                if (Age < 65.000000000) {
                    return -0.467891455;
                } else {
                    return 1.277148840;
                }
            } else {
                if (Age < 65.000000000) {
                    return 1.287432790;
                } else {
                    return -0.110862359;
                }
            }
        }
    }
}

inline double predict_tree_236(double Gender, double Age, double BMI) {
    if (BMI < 27.450605400) {
        if (BMI < 23.437500000) {
            if (Age < 75.000000000) {
                if (Age < 70.000000000) {
                    return -0.032783113;
                } else {
                    return 0.522272050;
                }
            } else {
                if (BMI < 22.829963700) {
                    return -0.803819120;
                } else {
                    return -0.113281421;
                }
            }
        } else {
            if (BMI < 23.634033200) {
                if (BMI < 23.597003900) {
                    return 0.097086221;
                } else {
                    return 0.688411832;
                }
            } else {
                if (Age < 66.000000000) {
                    return 0.014824549;
                } else {
                    return -0.060645208;
                }
            }
        }
    } else {
        if (Age < 77.000000000) {
            if (Age < 61.000000000) {
                if (Age < 58.000000000) {
                    return -0.276502728;
                } else {
                    return 0.797933578;
                }
            } else {
                if (Age < 67.000000000) {
                    return -1.160219790;
                } else {
                    return -0.316465586;
                }
            }
        } else {
            if (BMI < 27.472526600) {
                return 1.585669640;
            } else {
                if (Age < 78.000000000) {
                    return 1.117638350;
                } else {
                    return -0.223507136;
                }
            }
        }
    }
}

inline double predict_tree_237(double Gender, double Age, double BMI) {
    if (Age < 79.000000000) {
        if (Age < 74.000000000) {
            if (Age < 57.000000000) {
                if (Age < 56.000000000) {
                    return -0.019422971;
                } else {
                    return -0.147527009;
                }
            } else {
                if (Age < 60.000000000) {
                    return 0.118881248;
                } else {
                    return 0.007230673;
                }
            }
        } else {
            if (Age < 75.000000000) {
                if (Gender < 1.000000000) {
                    return -0.667376161;
                } else {
                    return 0.028756401;
                }
            } else {
                if (Age < 76.000000000) {
                    return 0.122361287;
                } else {
                    return -0.071278103;
                }
            }
        }
    } else {
        return 0.213351592;
    }
}

inline double predict_tree_238(double Gender, double Age, double BMI) {
    if (BMI < 24.772096600) {
        if (BMI < 24.740938200) {
            if (BMI < 24.271844900) {
                if (BMI < 24.121749900) {
                    return -0.004329980;
                } else {
                    return 0.193635389;
                }
            } else {
                if (BMI < 24.341758700) {
                    return -0.292782903;
                } else {
                    return -0.002714718;
                }
            }
        } else {
            if (Gender < 1.000000000) {
                return -0.214347124;
            } else {
                return -0.837740600;
            }
        }
    } else {
        if (BMI < 24.784257900) {
            if (Gender < 1.000000000) {
                return 0.766165912;
            } else {
                return 0.332403511;
            }
        } else {
            if (BMI < 25.057359700) {
                if (BMI < 24.967113500) {
                    return -0.010768461;
                } else {
                    return 0.259877682;
                }
            } else {
                if (BMI < 25.181078000) {
                    return -0.111876234;
                } else {
                    return 0.028459672;
                }
            }
        }
    }
}

inline double predict_tree_239(double Gender, double Age, double BMI) {
    if (Age < 67.000000000) {
        if (Gender < 1.000000000) {
            if (Age < 66.000000000) {
                if (Age < 49.000000000) {
                    return -0.046540797;
                } else {
                    return 0.060303606;
                }
            } else {
                return -0.469090492;
            }
        } else {
            if (Age < 66.000000000) {
                if (Age < 65.000000000) {
                    return 0.005677909;
                } else {
                    return -0.362706363;
                }
            } else {
                return 0.321315140;
            }
        }
    } else {
        if (Gender < 1.000000000) {
            if (Age < 71.000000000) {
                if (Age < 70.000000000) {
                    return 0.146120772;
                } else {
                    return 0.494483829;
                }
            } else {
                if (Age < 72.000000000) {
                    return -0.262300670;
                } else {
                    return 0.126144171;
                }
            }
        } else {
            if (Age < 73.000000000) {
                if (Age < 70.000000000) {
                    return 0.064255081;
                } else {
                    return -0.159851491;
                }
            } else {
                if (Age < 77.000000000) {
                    return 0.111458719;
                } else {
                    return -0.028801791;
                }
            }
        }
    }
}

inline double predict_tree_240(double Gender, double Age, double BMI) {
    if (Age < 36.000000000) {
        if (BMI < 24.111507400) {
            if (BMI < 24.034406700) {
                if (BMI < 24.017253900) {
                    return -0.011729526;
                } else {
                    return 0.456679493;
                }
            } else {
                if (Age < 25.000000000) {
                    return -0.960606754;
                } else {
                    return -0.228041396;
                }
            }
        } else {
            if (BMI < 24.251277900) {
                if (BMI < 24.178978000) {
                    return 0.673406482;
                } else {
                    return 0.314229548;
                }
            } else {
                if (Age < 27.000000000) {
                    return -0.046647165;
                } else {
                    return 0.089006744;
                }
            }
        }
    } else {
        if (BMI < 24.956596400) {
            if (BMI < 24.740938200) {
                if (Age < 45.000000000) {
                    return -0.102368772;
                } else {
                    return 0.010356745;
                }
            } else {
                if (Age < 62.000000000) {
                    return -0.470732599;
                } else {
                    return 0.121371560;
                }
            }
        } else {
            if (BMI < 25.280109400) {
                if (Age < 49.000000000) {
                    return -0.106522933;
                } else {
                    return 0.288497597;
                }
            } else {
                if (Age < 50.000000000) {
                    return 0.100304380;
                } else {
                    return -0.052758053;
                }
            }
        }
    }
}

inline double predict_tree_241(double Gender, double Age, double BMI) {
    if (Age < 78.000000000) {
        if (Age < 24.000000000) {
            if (Age < 21.000000000) {
                if (Gender < 1.000000000) {
                    return -0.051545281;
                } else {
                    return 0.000567042;
                }
            } else {
                if (Age < 23.000000000) {
                    return 0.061632760;
                } else {
                    return 0.008679362;
                }
            }
        } else {
            if (Age < 75.000000000) {
                if (Age < 25.000000000) {
                    return -0.080256321;
                } else {
                    return -0.014587552;
                }
            } else {
                if (Gender < 1.000000000) {
                    return 0.160727218;
                } else {
                    return -0.031309500;
                }
            }
        }
    } else {
        if (Gender < 1.000000000) {
            if (Age < 79.000000000) {
                return -0.461859584;
            } else {
                return -0.038058937;
            }
        } else {
            if (Age < 79.000000000) {
                return 0.034170669;
            } else {
                return -0.114181772;
            }
        }
    }
}

inline double predict_tree_242(double Gender, double Age, double BMI) {
    if (BMI < 21.952478400) {
        if (BMI < 21.913805000) {
            if (BMI < 21.773841900) {
                if (BMI < 21.545091600) {
                    return -0.046754666;
                } else {
                    return 0.329931855;
                }
            } else {
                if (BMI < 21.877550100) {
                    return -0.096084438;
                } else {
                    return -0.274631202;
                }
            }
        } else {
            return 0.588694930;
        }
    } else {
        if (BMI < 22.230987500) {
            if (BMI < 22.212966900) {
                if (BMI < 22.100290300) {
                    return -0.290076971;
                } else {
                    return -0.004846679;
                }
            } else {
                return -0.424987137;
            }
        } else {
            if (BMI < 22.476587300) {
                if (BMI < 22.408786800) {
                    return 0.044103045;
                } else {
                    return 0.384364694;
                }
            } else {
                if (BMI < 22.521507300) {
                    return -0.190872133;
                } else {
                    return -0.005591444;
                }
            }
        }
    }
}

inline double predict_tree_243(double Gender, double Age, double BMI) {
    if (Age < 77.000000000) {
        if (Age < 70.000000000) {
            if (Age < 41.000000000) {
                if (Age < 40.000000000) {
                    return -0.017129699;
                } else {
                    return -0.135399789;
                }
            } else {
                if (Age < 46.000000000) {
                    return 0.095937640;
                } else {
                    return -0.014107518;
                }
            }
        } else {
            if (Gender < 1.000000000) {
                if (Age < 71.000000000) {
                    return 0.232927933;
                } else {
                    return -0.048490763;
                }
            } else {
                if (Age < 71.000000000) {
                    return -0.460596830;
                } else {
                    return -0.095345102;
                }
            }
        }
    } else {
        if (Gender < 1.000000000) {
            if (Age < 78.000000000) {
                return 0.287777096;
            } else {
                if (Age < 79.000000000) {
                    return 0.053837542;
                } else {
                    return 0.178659230;
                }
            }
        } else {
            if (Age < 78.000000000) {
                return -0.201850995;
            } else {
                if (Age < 79.000000000) {
                    return 0.022420073;
                } else {
                    return -0.039823614;
                }
            }
        }
    }
}

inline double predict_tree_244(double Gender, double Age, double BMI) {
    if (BMI < 23.893259000) {
        if (BMI < 23.849777200) {
            if (BMI < 23.808691000) {
                if (BMI < 23.597003900) {
                    return 0.003970339;
                } else {
                    return 0.157302290;
                }
            } else {
                if (BMI < 23.833004000) {
                    return -0.517982423;
                } else {
                    return -0.055440579;
                }
            }
        } else {
            if (Gender < 1.000000000) {
                return 0.691066802;
            } else {
                return 0.338454127;
            }
        }
    } else {
        if (BMI < 24.017253900) {
            if (BMI < 23.936061900) {
                if (Gender < 1.000000000) {
                    return -0.261097431;
                } else {
                    return 0.112857737;
                }
            } else {
                if (BMI < 23.939481700) {
                    return -0.767556489;
                } else {
                    return -0.159185171;
                }
            }
        } else {
            if (BMI < 24.034406700) {
                if (Gender < 1.000000000) {
                    return 0.808596015;
                } else {
                    return -0.085654192;
                }
            } else {
                if (BMI < 24.304616900) {
                    return 0.063487336;
                } else {
                    return -0.013222930;
                }
            }
        }
    }
}

inline double predict_tree_245(double Gender, double Age, double BMI) {
    if (Age < 66.000000000) {
        if (Age < 65.000000000) {
            if (Age < 46.000000000) {
                if (Age < 41.000000000) {
                    return 0.000022388;
                } else {
                    return 0.100543588;
                }
            } else {
                if (Age < 57.000000000) {
                    return -0.049648154;
                } else {
                    return 0.041420631;
                }
            }
        } else {
            if (Gender < 1.000000000) {
                return 0.200580567;
            } else {
                return -0.456254303;
            }
        }
    } else {
        if (Age < 68.000000000) {
            if (Age < 67.000000000) {
                if (Gender < 1.000000000) {
                    return -0.337180525;
                } else {
                    return 0.339082569;
                }
            } else {
                if (Gender < 1.000000000) {
                    return 0.581983328;
                } else {
                    return -0.021596730;
                }
            }
        } else {
            if (Age < 73.000000000) {
                if (Age < 70.000000000) {
                    return 0.047335118;
                } else {
                    return -0.032917041;
                }
            } else {
                if (Gender < 1.000000000) {
                    return 0.109618358;
                } else {
                    return 0.063831478;
                }
            }
        }
    }
}

inline double predict_tree_246(double Gender, double Age, double BMI) {
    if (BMI < 23.795360600) {
        if (BMI < 23.597003900) {
            if (Age < 55.000000000) {
                if (BMI < 22.862533600) {
                    return 0.020586600;
                } else {
                    return -0.103667654;
                }
            } else {
                if (Age < 60.000000000) {
                    return 0.372851044;
                } else {
                    return 0.031643711;
                }
            }
        } else {
            if (BMI < 23.634033200) {
                if (Age < 70.000000000) {
                    return 0.747877002;
                } else {
                    return -0.528851807;
                }
            } else {
                if (Age < 55.000000000) {
                    return 0.207485124;
                } else {
                    return -0.309488893;
                }
            }
        }
    } else {
        if (BMI < 23.849777200) {
            if (Age < 61.000000000) {
                if (Age < 46.000000000) {
                    return -0.424827158;
                } else {
                    return 0.751565397;
                }
            } else {
                if (BMI < 23.833004000) {
                    return -1.101449370;
                } else {
                    return 0.198625371;
                }
            }
        } else {
            if (BMI < 23.893259000) {
                if (Age < 42.000000000) {
                    return 0.009569491;
                } else {
                    return 0.617136598;
                }
            } else {
                if (BMI < 23.939481700) {
                    return -0.261235356;
                } else {
                    return -0.011759452;
                }
            }
        }
    }
}

inline double predict_tree_247(double Gender, double Age, double BMI) {
    if (Age < 79.000000000) {
        if (BMI < 22.837369900) {
            if (Age < 75.000000000) {
                if (Age < 71.000000000) {
                    return 0.064057402;
                } else {
                    return 0.771689236;
                }
            } else {
                if (BMI < 22.769437800) {
                    return -0.897136509;
                } else {
                    return 1.146482110;
                }
            }
        } else {
            if (BMI < 22.981901200) {
                if (Age < 35.000000000) {
                    return -0.440714359;
                } else {
                    return 0.175676659;
                }
            } else {
                if (BMI < 23.147254900) {
                    return 0.145538151;
                } else {
                    return -0.012475585;
                }
            }
        }
    } else {
        if (BMI < 25.977846100) {
            if (BMI < 21.303949400) {
                return 0.686935008;
            } else {
                if (BMI < 23.722810700) {
                    return -1.015763640;
                } else {
                    return -0.359477192;
                }
            }
        } else {
            if (BMI < 26.485540400) {
                if (BMI < 26.315052000) {
                    return 1.851961140;
                } else {
                    return 0.483962953;
                }
            } else {
                if (BMI < 26.827421200) {
                    return -1.376092080;
                } else {
                    return 0.337454259;
                }
            }
        }
    }
}

inline double predict_tree_248(double Gender, double Age, double BMI) {
    if (BMI < 20.796730000) {
        if (Age < 24.000000000) {
            if (Age < 21.000000000) {
                if (BMI < 20.504934300) {
                    return 0.491207182;
                } else {
                    return -0.078750610;
                }
            } else {
                return -1.103388550;
            }
        } else {
            if (Age < 26.000000000) {
                return 0.763252914;
            } else {
                if (Age < 29.000000000) {
                    return -0.868263364;
                } else {
                    return -0.004363616;
                }
            }
        }
    } else {
        if (BMI < 21.218317000) {
            if (Age < 46.000000000) {
                if (Age < 21.000000000) {
                    return -0.365783423;
                } else {
                    return 0.458926946;
                }
            } else {
                if (Age < 65.000000000) {
                    return -0.610240817;
                } else {
                    return 0.384817958;
                }
            }
        } else {
            if (BMI < 21.545091600) {
                if (Age < 47.000000000) {
                    return -0.263466090;
                } else {
                    return 0.531568825;
                }
            } else {
                if (BMI < 24.419279100) {
                    return 0.022727381;
                } else {
                    return -0.024224060;
                }
            }
        }
    }
}

inline double predict_tree_249(double Gender, double Age, double BMI) {
    if (BMI < 23.808691000) {
        if (BMI < 23.597003900) {
            if (Age < 27.000000000) {
                if (Age < 22.000000000) {
                    return -0.078797467;
                } else {
                    return 0.175152197;
                }
            } else {
                if (BMI < 23.555555300) {
                    return -0.016007181;
                } else {
                    return -0.432139456;
                }
            }
        } else {
            if (Age < 23.000000000) {
                if (BMI < 23.733238200) {
                    return 0.260168672;
                } else {
                    return -1.407132860;
                }
            } else {
                if (Age < 55.000000000) {
                    return 0.352361143;
                } else {
                    return -0.131608486;
                }
            }
        }
    } else {
        if (BMI < 23.849777200) {
            if (Age < 23.000000000) {
                if (BMI < 23.808797800) {
                    return -0.045219269;
                } else {
                    return 0.637213767;
                }
            } else {
                if (BMI < 23.808797800) {
                    return -0.039514549;
                } else {
                    return -0.645974159;
                }
            }
        } else {
            if (BMI < 23.893259000) {
                if (Age < 28.000000000) {
                    return -0.267965019;
                } else {
                    return 0.430536687;
                }
            } else {
                if (BMI < 23.939481700) {
                    return -0.332440495;
                } else {
                    return -0.002775176;
                }
            }
        }
    }
}

inline double predict_tree_250(double Gender, double Age, double BMI) {
    if (Age < 57.000000000) {
        if (BMI < 27.160493900) {
            if (BMI < 26.880950900) {
                if (BMI < 26.869806300) {
                    return -0.002626629;
                } else {
                    return -1.033984180;
                }
            } else {
                if (Age < 53.000000000) {
                    return 0.250710130;
                } else {
                    return 1.300410150;
                }
            }
        } else {
            if (Age < 55.000000000) {
                if (BMI < 27.776929900) {
                    return -0.260885656;
                } else {
                    return 0.373342514;
                }
            } else {
                if (BMI < 27.472526600) {
                    return -1.215307950;
                } else {
                    return -0.410167813;
                }
            }
        }
    } else {
        if (BMI < 21.469150500) {
            if (Age < 71.000000000) {
                if (BMI < 20.957170500) {
                    return 0.628935575;
                } else {
                    return -0.375624776;
                }
            } else {
                return 1.257765050;
            }
        } else {
            if (BMI < 21.913805000) {
                if (BMI < 21.773841900) {
                    return -0.233080059;
                } else {
                    return -0.833150625;
                }
            } else {
                if (Age < 58.000000000) {
                    return 0.262508482;
                } else {
                    return 0.019357013;
                }
            }
        }
    }
}

inline double predict_tree_251(double Gender, double Age, double BMI) {
    if (BMI < 27.776929900) {
        if (BMI < 25.420015300) {
            if (Age < 77.000000000) {
                if (Age < 42.000000000) {
                    return 0.033918675;
                } else {
                    return -0.023116128;
                }
            } else {
                if (BMI < 23.833004000) {
                    return -0.065034442;
                } else {
                    return 0.427693516;
                }
            }
        } else {
            if (Age < 31.000000000) {
                if (BMI < 25.469387100) {
                    return -0.519158900;
                } else {
                    return 0.144475430;
                }
            } else {
                if (Age < 39.000000000) {
                    return -0.254092723;
                } else {
                    return -0.018847406;
                }
            }
        }
    } else {
        if (Age < 61.000000000) {
            if (Age < 40.000000000) {
                if (Age < 29.000000000) {
                    return 0.263505191;
                } else {
                    return -1.022319440;
                }
            } else {
                if (Age < 53.000000000) {
                    return 1.568747880;
                } else {
                    return 0.533030212;
                }
            }
        } else {
            if (Age < 65.000000000) {
                return -1.624246840;
            } else {
                if (Age < 79.000000000) {
                    return 0.232316047;
                } else {
                    return 1.532969950;
                }
            }
        }
    }
}

inline double predict_tree_252(double Gender, double Age, double BMI) {
    if (BMI < 21.829952200) {
        if (Age < 34.000000000) {
            if (Age < 29.000000000) {
                if (Age < 25.000000000) {
                    return 0.015134165;
                } else {
                    return 0.357359141;
                }
            } else {
                if (BMI < 21.303949400) {
                    return -0.123572372;
                } else {
                    return -0.574350357;
                }
            }
        } else {
            if (BMI < 21.718065300) {
                if (Age < 45.000000000) {
                    return 0.314310402;
                } else {
                    return -0.206399277;
                }
            } else {
                if (Age < 55.000000000) {
                    return 0.547365308;
                } else {
                    return 2.080912350;
                }
            }
        }
    } else {
        if (BMI < 22.230987500) {
            if (Age < 68.000000000) {
                if (Age < 60.000000000) {
                    return -0.059960984;
                } else {
                    return -0.659717321;
                }
            } else {
                if (Age < 75.000000000) {
                    return 1.109140040;
                } else {
                    return -0.515979052;
                }
            }
        } else {
            if (BMI < 22.347782100) {
                if (Age < 59.000000000) {
                    return 0.415291697;
                } else {
                    return -0.693607032;
                }
            } else {
                if (BMI < 22.375679000) {
                    return -0.659772575;
                } else {
                    return 0.015250836;
                }
            }
        }
    }
}

inline double predict_tree_253(double Gender, double Age, double BMI) {
    if (BMI < 24.251277900) {
        if (BMI < 24.121749900) {
            if (BMI < 24.088222500) {
                if (BMI < 24.023809400) {
                    return 0.019986503;
                } else {
                    return 0.309939384;
                }
            } else {
                if (BMI < 24.111507400) {
                    return -0.550465465;
                } else {
                    return -0.017564274;
                }
            }
        } else {
            if (BMI < 24.158817300) {
                return 0.630431950;
            } else {
                if (Gender < 1.000000000) {
                    return 0.276924729;
                } else {
                    return -0.103339300;
                }
            }
        }
    } else {
        if (BMI < 24.772096600) {
            if (BMI < 24.725183500) {
                if (BMI < 24.670375800) {
                    return -0.090661325;
                } else {
                    return 0.191297427;
                }
            } else {
                if (BMI < 24.740938200) {
                    return -0.263281852;
                } else {
                    return -0.475209445;
                }
            }
        } else {
            if (BMI < 24.835645700) {
                if (Gender < 1.000000000) {
                    return 0.230003238;
                } else {
                    return 0.877210319;
                }
            } else {
                if (Gender < 1.000000000) {
                    return 0.015644116;
                } else {
                    return -0.113643512;
                }
            }
        }
    }
}

inline double predict_tree_254(double Gender, double Age, double BMI) {
    if (BMI < 25.306932400) {
        if (BMI < 25.207756000) {
            if (Age < 77.000000000) {
                if (BMI < 25.102392200) {
                    return 0.013628935;
                } else {
                    return -0.199453190;
                }
            } else {
                if (BMI < 25.029760400) {
                    return 0.056163549;
                } else {
                    return 1.543541190;
                }
            }
        } else {
            if (Age < 77.000000000) {
                if (Age < 59.000000000) {
                    return 0.064409181;
                } else {
                    return 0.896505237;
                }
            } else {
                return -1.104923010;
            }
        }
    } else {
        if (Age < 51.000000000) {
            if (Age < 39.000000000) {
                if (BMI < 25.951557200) {
                    return -0.151239276;
                } else {
                    return 0.087691121;
                }
            } else {
                if (BMI < 25.854639100) {
                    return 0.350082844;
                } else {
                    return 0.055307373;
                }
            }
        } else {
            if (Age < 53.000000000) {
                if (BMI < 27.776929900) {
                    return -0.656867743;
                } else {
                    return 1.450755240;
                }
            } else {
                if (BMI < 26.555965400) {
                    return -0.171224132;
                } else {
                    return 0.069519311;
                }
            }
        }
    }
}

inline double predict_tree_255(double Gender, double Age, double BMI) {
    if (Age < 54.000000000) {
        if (Age < 51.000000000) {
            if (Age < 45.000000000) {
                if (Age < 36.000000000) {
                    return -0.003698726;
                } else {
                    return -0.041421548;
                }
            } else {
                if (Age < 48.000000000) {
                    return 0.076860003;
                } else {
                    return -0.033995144;
                }
            }
        } else {
            if (Gender < 1.000000000) {
                if (Age < 52.000000000) {
                    return -0.339918673;
                } else {
                    return -0.028124940;
                }
            } else {
                if (Age < 52.000000000) {
                    return 0.196248755;
                } else {
                    return -0.157814875;
                }
            }
        }
    } else {
        if (Age < 59.000000000) {
            if (Age < 57.000000000) {
                if (Age < 56.000000000) {
                    return 0.083561495;
                } else {
                    return -0.158112556;
                }
            } else {
                if (Age < 58.000000000) {
                    return 0.161200419;
                } else {
                    return 0.247105271;
                }
            }
        } else {
            if (Age < 65.000000000) {
                if (Age < 64.000000000) {
                    return -0.009591547;
                } else {
                    return 0.221461996;
                }
            } else {
                if (Age < 72.000000000) {
                    return -0.054717023;
                } else {
                    return 0.023311695;
                }
            }
        }
    }
}

inline double predict_tree_256(double Gender, double Age, double BMI) {
    if (Age < 41.000000000) {
        if (Age < 40.000000000) {
            if (Age < 39.000000000) {
                if (Age < 38.000000000) {
                    return 0.003785121;
                } else {
                    return -0.144580513;
                }
            } else {
                if (Gender < 1.000000000) {
                    return 0.048505276;
                } else {
                    return 0.183128044;
                }
            }
        } else {
            if (Gender < 1.000000000) {
                return -0.086951874;
            } else {
                return -0.174198091;
            }
        }
    } else {
        if (Gender < 1.000000000) {
            if (Age < 44.000000000) {
                if (Age < 43.000000000) {
                    return 0.117512703;
                } else {
                    return 0.417092085;
                }
            } else {
                if (Age < 45.000000000) {
                    return -0.206260696;
                } else {
                    return 0.080256797;
                }
            }
        } else {
            if (Age < 56.000000000) {
                if (Age < 44.000000000) {
                    return -0.106028423;
                } else {
                    return 0.081801824;
                }
            } else {
                if (Age < 57.000000000) {
                    return -0.399720401;
                } else {
                    return -0.043095980;
                }
            }
        }
    }
}

inline double predict_tree_257(double Gender, double Age, double BMI) {
    if (BMI < 20.504934300) {
        if (Age < 21.000000000) {
            return 0.429183751;
        } else {
            if (Age < 29.000000000) {
                if (Age < 24.000000000) {
                    return -1.108024950;
                } else {
                    return -0.481873453;
                }
            } else {
                if (Age < 33.000000000) {
                    return 0.789515138;
                } else {
                    return -0.123483047;
                }
            }
        }
    } else {
        if (BMI < 22.491350200) {
            if (Age < 22.000000000) {
                if (BMI < 22.282478300) {
                    return -0.387698293;
                } else {
                    return 0.504583895;
                }
            } else {
                if (Age < 58.000000000) {
                    return 0.051045168;
                } else {
                    return 0.267481446;
                }
            }
        } else {
            if (BMI < 22.546575500) {
                if (Age < 59.000000000) {
                    return -0.181427002;
                } else {
                    return -1.160056110;
                }
            } else {
                if (BMI < 23.597003900) {
                    return -0.038776781;
                } else {
                    return 0.017436080;
                }
            }
        }
    }
}

inline double predict_tree_258(double Gender, double Age, double BMI) {
    if (BMI < 22.862533600) {
        if (Age < 60.000000000) {
            if (Age < 58.000000000) {
                if (BMI < 22.230987500) {
                    return -0.017276079;
                } else {
                    return 0.145238087;
                }
            } else {
                if (BMI < 21.989893000) {
                    return 0.158209801;
                } else {
                    return 1.472800610;
                }
            }
        } else {
            if (Age < 65.000000000) {
                if (BMI < 22.662708300) {
                    return -0.471719056;
                } else {
                    return -1.288525460;
                }
            } else {
                if (Age < 75.000000000) {
                    return 0.139076456;
                } else {
                    return -0.576445937;
                }
            }
        }
    } else {
        if (BMI < 22.981901200) {
            if (Age < 47.000000000) {
                if (Age < 35.000000000) {
                    return -0.667188048;
                } else {
                    return -0.104187027;
                }
            } else {
                if (BMI < 22.948116300) {
                    return 0.767512739;
                } else {
                    return -0.399424881;
                }
            }
        } else {
            if (BMI < 23.147254900) {
                if (Age < 65.000000000) {
                    return 0.014845510;
                } else {
                    return 0.804669797;
                }
            } else {
                if (BMI < 23.225431400) {
                    return -0.443338186;
                } else {
                    return -0.014715242;
                }
            }
        }
    }
}

inline double predict_tree_259(double Gender, double Age, double BMI) {
    if (BMI < 27.084159900) {
        if (BMI < 26.794937100) {
            if (BMI < 21.367521300) {
                if (BMI < 21.303949400) {
                    return 0.079483323;
                } else {
                    return 0.523173332;
                }
            } else {
                if (BMI < 22.100290300) {
                    return -0.074454509;
                } else {
                    return 0.008597394;
                }
            }
        } else {
            if (Gender < 1.000000000) {
                if (BMI < 26.869806300) {
                    return 0.249719352;
                } else {
                    return 0.036339369;
                }
            } else {
                return 1.024191500;
            }
        }
    } else {
        if (BMI < 27.714790300) {
            if (BMI < 27.137414900) {
                return -0.464762747;
            } else {
                if (BMI < 27.160493900) {
                    return -0.054585144;
                } else {
                    return -0.216712236;
                }
            }
        } else {
            return 0.158421546;
        }
    }
}

inline double predict_tree_260(double Gender, double Age, double BMI) {
    if (BMI < 24.441803000) {
        if (BMI < 24.121749900) {
            if (BMI < 23.922422400) {
                if (BMI < 23.849777200) {
                    return 0.005517713;
                } else {
                    return 0.228696749;
                }
            } else {
                if (BMI < 23.999458300) {
                    return -0.404222131;
                } else {
                    return -0.010651344;
                }
            }
        } else {
            if (BMI < 24.163265200) {
                if (BMI < 24.141519500) {
                    return 0.231910199;
                } else {
                    return 0.416268706;
                }
            } else {
                if (BMI < 24.212293600) {
                    return -0.323551327;
                } else {
                    return 0.119877353;
                }
            }
        }
    } else {
        if (BMI < 24.456064200) {
            if (Gender < 1.000000000) {
                if (BMI < 24.447279000) {
                    return 0.030815246;
                } else {
                    return -0.319498032;
                }
            } else {
                if (BMI < 24.447279000) {
                    return -0.911925077;
                } else {
                    return -0.198621362;
                }
            }
        } else {
            if (BMI < 24.956596400) {
                if (BMI < 24.913494100) {
                    return -0.032530799;
                } else {
                    return -0.559624553;
                }
            } else {
                if (BMI < 25.057359700) {
                    return 0.174827412;
                } else {
                    return -0.022101853;
                }
            }
        }
    }
}

inline double predict_tree_261(double Gender, double Age, double BMI) {
    if (BMI < 26.282564200) {
        if (BMI < 26.078971900) {
            if (BMI < 26.037494700) {
                if (BMI < 26.023048400) {
                    return -0.001592695;
                } else {
                    return 0.196073979;
                }
            } else {
                if (Gender < 1.000000000) {
                    return -0.409702420;
                } else {
                    return 0.951612532;
                }
            }
        } else {
            if (BMI < 26.196188000) {
                if (BMI < 26.155626300) {
                    return 0.292481333;
                } else {
                    return 0.036418386;
                }
            } else {
                if (Gender < 1.000000000) {
                    return 0.367544472;
                } else {
                    return -0.273263931;
                }
            }
        }
    } else {
        if (BMI < 26.296566000) {
            if (Gender < 1.000000000) {
                return -0.625407994;
            } else {
                return 0.061208647;
            }
        } else {
            if (BMI < 26.303619400) {
                return 0.323921353;
            } else {
                if (BMI < 26.312810900) {
                    return -0.453395039;
                } else {
                    return -0.027051024;
                }
            }
        }
    }
}

inline double predict_tree_262(double Gender, double Age, double BMI) {
    if (Age < 79.000000000) {
        if (Gender < 1.000000000) {
            if (Age < 21.000000000) {
                return -0.188552663;
            } else {
                if (Age < 70.000000000) {
                    return 0.004186086;
                } else {
                    return 0.069062553;
                }
            }
        } else {
            if (Age < 52.000000000) {
                if (Age < 51.000000000) {
                    return -0.020766439;
                } else {
                    return 0.241575867;
                }
            } else {
                if (Age < 63.000000000) {
                    return -0.140837237;
                } else {
                    return -0.011122235;
                }
            }
        }
    } else {
        if (Gender < 1.000000000) {
            return -0.371328473;
        } else {
            return -0.103742100;
        }
    }
}

inline double predict_tree_263(double Gender, double Age, double BMI) {
    if (Age < 28.000000000) {
        if (BMI < 25.469387100) {
            if (BMI < 25.227865200) {
                if (Age < 26.000000000) {
                    return 0.012382330;
                } else {
                    return -0.173339814;
                }
            } else {
                if (BMI < 25.351541500) {
                    return -0.339765429;
                } else {
                    return -0.721594274;
                }
            }
        } else {
            if (Age < 26.000000000) {
                if (Age < 23.000000000) {
                    return 0.194158867;
                } else {
                    return -0.121797509;
                }
            } else {
                if (BMI < 26.234567600) {
                    return 0.034298785;
                } else {
                    return 0.710038483;
                }
            }
        }
    } else {
        if (BMI < 25.420015300) {
            if (BMI < 25.227865200) {
                if (Age < 76.000000000) {
                    return 0.020360529;
                } else {
                    return 0.243016034;
                }
            } else {
                if (Age < 33.000000000) {
                    return 0.841002047;
                } else {
                    return 0.107055172;
                }
            }
        } else {
            if (BMI < 25.767219500) {
                if (Age < 78.000000000) {
                    return -0.181576818;
                } else {
                    return 0.594388068;
                }
            } else {
                if (Age < 77.000000000) {
                    return 0.052389119;
                } else {
                    return -0.403833836;
                }
            }
        }
    }
}

inline double predict_tree_264(double Gender, double Age, double BMI) {
    if (Age < 78.000000000) {
        if (Age < 77.000000000) {
            if (Age < 73.000000000) {
                if (Age < 42.000000000) {
                    return 0.005769858;
                } else {
                    return -0.016133189;
                }
            } else {
                if (Age < 74.000000000) {
                    return 0.125229686;
                } else {
                    return 0.009347219;
                }
            }
        } else {
            if (Gender < 1.000000000) {
                return 0.379394025;
            } else {
                return -0.057404593;
            }
        }
    } else {
        if (Gender < 1.000000000) {
            if (Age < 79.000000000) {
                return 0.021965448;
            } else {
                return -0.142089516;
            }
        } else {
            if (Age < 79.000000000) {
                return -0.208506614;
            } else {
                return -0.102685541;
            }
        }
    }
}

inline double predict_tree_265(double Gender, double Age, double BMI) {
    if (Age < 54.000000000) {
        if (Age < 52.000000000) {
            if (Age < 49.000000000) {
                if (Age < 48.000000000) {
                    return -0.001203895;
                } else {
                    return -0.150202379;
                }
            } else {
                if (Age < 50.000000000) {
                    return 0.224876970;
                } else {
                    return 0.040718920;
                }
            }
        } else {
            if (Gender < 1.000000000) {
                if (Age < 53.000000000) {
                    return -0.127748355;
                } else {
                    return -0.246509477;
                }
            } else {
                return -0.132870868;
            }
        }
    } else {
        if (Gender < 1.000000000) {
            if (Age < 55.000000000) {
                return 0.732331097;
            } else {
                if (Age < 56.000000000) {
                    return -0.238237992;
                } else {
                    return 0.069675595;
                }
            }
        } else {
            if (Age < 57.000000000) {
                if (Age < 56.000000000) {
                    return -0.035182491;
                } else {
                    return -0.312264293;
                }
            } else {
                if (Age < 60.000000000) {
                    return 0.092265233;
                } else {
                    return -0.031310398;
                }
            }
        }
    }
}

inline double predict_tree_266(double Gender, double Age, double BMI) {
    if (BMI < 27.714790300) {
        if (BMI < 23.423557300) {
            if (BMI < 23.374725300) {
                if (Gender < 1.000000000) {
                    return -0.223462716;
                } else {
                    return -0.007875011;
                }
            } else {
                if (BMI < 23.388687100) {
                    return -0.538132071;
                } else {
                    return -0.194194898;
                }
            }
        } else {
            if (BMI < 23.795360600) {
                if (BMI < 23.772178600) {
                    return 0.107268162;
                } else {
                    return 0.602922797;
                }
            } else {
                if (BMI < 23.833004000) {
                    return -0.298166931;
                } else {
                    return -0.000006986;
                }
            }
        }
    } else {
        return 0.330474585;
    }
}

inline double predict_tree_267(double Gender, double Age, double BMI) {
    if (Age < 79.000000000) {
        if (Age < 42.000000000) {
            if (Age < 41.000000000) {
                if (Gender < 1.000000000) {
                    return -0.030144134;
                } else {
                    return 0.000361869;
                }
            } else {
                if (Gender < 1.000000000) {
                    return -0.046413310;
                } else {
                    return 0.246512219;
                }
            }
        } else {
            if (Age < 44.000000000) {
                if (Gender < 1.000000000) {
                    return -0.035838298;
                } else {
                    return -0.216856852;
                }
            } else {
                if (Age < 52.000000000) {
                    return -0.000643631;
                } else {
                    return -0.040412787;
                }
            }
        }
    } else {
        if (Gender < 1.000000000) {
            return 0.310923219;
        } else {
            return -0.040862460;
        }
    }
}

inline double predict_tree_268(double Gender, double Age, double BMI) {
    if (Age < 24.000000000) {
        if (Age < 22.000000000) {
            if (Gender < 1.000000000) {
                if (Age < 21.000000000) {
                    return 0.023239076;
                } else {
                    return -0.074191190;
                }
            } else {
                if (Age < 21.000000000) {
                    return -0.007435137;
                } else {
                    return 0.078959532;
                }
            }
        } else {
            if (Gender < 1.000000000) {
                if (Age < 23.000000000) {
                    return 0.070825152;
                } else {
                    return 0.140687719;
                }
            } else {
                if (Age < 23.000000000) {
                    return 0.076249458;
                } else {
                    return 0.006185477;
                }
            }
        }
    } else {
        if (Age < 28.000000000) {
            if (Age < 27.000000000) {
                if (Age < 26.000000000) {
                    return -0.068739750;
                } else {
                    return -0.003362983;
                }
            } else {
                if (Gender < 1.000000000) {
                    return -0.050857537;
                } else {
                    return -0.148437843;
                }
            }
        } else {
            if (Age < 30.000000000) {
                if (Gender < 1.000000000) {
                    return 0.046607990;
                } else {
                    return 0.117891051;
                }
            } else {
                if (Age < 70.000000000) {
                    return -0.011382976;
                } else {
                    return 0.029763542;
                }
            }
        }
    }
}

inline double predict_tree_269(double Gender, double Age, double BMI) {
    if (BMI < 24.017253900) {
        if (BMI < 23.936061900) {
            if (BMI < 23.147254900) {
                if (BMI < 23.082542400) {
                    return 0.007242622;
                } else {
                    return 0.403679729;
                }
            } else {
                if (BMI < 23.225431400) {
                    return -0.448866338;
                } else {
                    return -0.010083543;
                }
            }
        } else {
            if (BMI < 23.939481700) {
                if (Gender < 1.000000000) {
                    return -0.103971757;
                } else {
                    return -0.775170207;
                }
            } else {
                if (BMI < 23.951227200) {
                    return -0.018108562;
                } else {
                    return -0.286611676;
                }
            }
        }
    } else {
        if (BMI < 24.088222500) {
            if (Gender < 1.000000000) {
                return 0.726862848;
            } else {
                if (BMI < 24.023809400) {
                    return -0.891963542;
                } else {
                    return -0.108393006;
                }
            }
        } else {
            if (BMI < 24.111507400) {
                if (Gender < 1.000000000) {
                    return -0.955360234;
                } else {
                    return -0.195519030;
                }
            } else {
                if (BMI < 24.163265200) {
                    return 0.285154521;
                } else {
                    return 0.018270649;
                }
            }
        }
    }
}

inline double predict_tree_270(double Gender, double Age, double BMI) {
    if (BMI < 26.485540400) {
        if (BMI < 26.078971900) {
            if (Age < 56.000000000) {
                if (BMI < 25.737081500) {
                    return -0.003401370;
                } else {
                    return 0.107575521;
                }
            } else {
                if (BMI < 25.306932400) {
                    return -0.014960568;
                } else {
                    return -0.180581942;
                }
            }
        } else {
            if (Age < 76.000000000) {
                if (Age < 26.000000000) {
                    return -0.374285817;
                } else {
                    return 0.265236676;
                }
            } else {
                if (BMI < 26.332889600) {
                    return -1.474341510;
                } else {
                    return 0.426522106;
                }
            }
        }
    } else {
        if (Age < 72.000000000) {
            if (Age < 62.000000000) {
                if (Age < 61.000000000) {
                    return -0.076692074;
                } else {
                    return 0.528553665;
                }
            } else {
                if (BMI < 27.714790300) {
                    return -0.339450866;
                } else {
                    return -1.022043470;
                }
            }
        } else {
            if (BMI < 26.592670400) {
                if (Age < 76.000000000) {
                    return -0.943046391;
                } else {
                    return -0.240793660;
                }
            } else {
                if (BMI < 26.713068000) {
                    return 1.437543150;
                } else {
                    return 0.128780529;
                }
            }
        }
    }
}

inline double predict_tree_271(double Gender, double Age, double BMI) {
    if (BMI < 23.808691000) {
        if (BMI < 23.711845400) {
            if (Age < 39.000000000) {
                if (Age < 26.000000000) {
                    return 0.064656690;
                } else {
                    return -0.094073236;
                }
            } else {
                if (Age < 60.000000000) {
                    return 0.130275160;
                } else {
                    return -0.051152829;
                }
            }
        } else {
            if (Age < 58.000000000) {
                if (Age < 23.000000000) {
                    return -0.572690666;
                } else {
                    return 0.604630053;
                }
            } else {
                if (Age < 67.000000000) {
                    return -0.840342641;
                } else {
                    return 0.161066741;
                }
            }
        }
    } else {
        if (BMI < 23.833004000) {
            if (Age < 61.000000000) {
                if (Age < 50.000000000) {
                    return -0.472291023;
                } else {
                    return 0.873142242;
                }
            } else {
                if (BMI < 23.808797800) {
                    return -0.041015282;
                } else {
                    return -1.707925200;
                }
            }
        } else {
            if (Age < 77.000000000) {
                if (BMI < 24.017253900) {
                    return -0.124908686;
                } else {
                    return 0.000725129;
                }
            } else {
                if (BMI < 24.323228800) {
                    return 1.310330510;
                } else {
                    return 0.020599062;
                }
            }
        }
    }
}

inline double predict_tree_272(double Gender, double Age, double BMI) {
    if (BMI < 24.618103000) {
        if (BMI < 24.609375000) {
            if (Age < 76.000000000) {
                if (Age < 64.000000000) {
                    return -0.013829352;
                } else {
                    return -0.129162505;
                }
            } else {
                if (BMI < 23.833004000) {
                    return -0.180272415;
                } else {
                    return 0.970738351;
                }
            }
        } else {
            if (Age < 48.000000000) {
                if (Age < 24.000000000) {
                    return -0.010953369;
                } else {
                    return -1.326994180;
                }
            } else {
                if (Age < 51.000000000) {
                    return 1.086545830;
                } else {
                    return -0.626380920;
                }
            }
        }
    } else {
        if (Age < 74.000000000) {
            if (BMI < 24.677021000) {
                if (BMI < 24.654832800) {
                    return 0.044781156;
                } else {
                    return 0.577784121;
                }
            } else {
                if (Age < 61.000000000) {
                    return -0.002966480;
                } else {
                    return 0.115640886;
                }
            }
        } else {
            if (BMI < 24.956596400) {
                if (BMI < 24.654832800) {
                    return 0.570068061;
                } else {
                    return -0.961920917;
                }
            } else {
                if (BMI < 25.469387100) {
                    return 0.485594958;
                } else {
                    return -0.285287440;
                }
            }
        }
    }
}

inline double predict_tree_273(double Gender, double Age, double BMI) {
    if (BMI < 24.560325600) {
        if (BMI < 24.456064200) {
            if (BMI < 24.441803000) {
                if (BMI < 24.141519500) {
                    return -0.005785097;
                } else {
                    return 0.111482419;
                }
            } else {
                if (Gender < 1.000000000) {
                    return -0.051577724;
                } else {
                    return -0.825781763;
                }
            }
        } else {
            if (BMI < 24.464601500) {
                return 1.058794500;
            } else {
                if (BMI < 24.515596400) {
                    return 0.038043097;
                } else {
                    return 0.278166771;
                }
            }
        }
    } else {
        if (BMI < 24.618103000) {
            if (BMI < 24.609375000) {
                if (Gender < 1.000000000) {
                    return -0.034444146;
                } else {
                    return -0.190835878;
                }
            } else {
                return -0.723351181;
            }
        } else {
            if (BMI < 24.635368300) {
                return 0.341155857;
            } else {
                if (BMI < 24.654832800) {
                    return -0.294054717;
                } else {
                    return -0.011758760;
                }
            }
        }
    }
}

inline double predict_tree_274(double Gender, double Age, double BMI) {
    if (BMI < 23.808691000) {
        if (BMI < 23.711845400) {
            if (BMI < 23.665245100) {
                if (BMI < 23.597003900) {
                    return 0.014939944;
                } else {
                    return 0.285152763;
                }
            } else {
                if (Age < 58.000000000) {
                    return -0.531349301;
                } else {
                    return 0.870660543;
                }
            }
        } else {
            if (Age < 58.000000000) {
                if (Age < 23.000000000) {
                    return -0.501108050;
                } else {
                    return 0.487315357;
                }
            } else {
                if (Age < 67.000000000) {
                    return -0.825305164;
                } else {
                    return 0.061465420;
                }
            }
        }
    } else {
        if (BMI < 23.833004000) {
            if (Age < 23.000000000) {
                if (BMI < 23.808797800) {
                    return -0.038414154;
                } else {
                    return 1.176163550;
                }
            } else {
                if (Age < 61.000000000) {
                    return -0.335840374;
                } else {
                    return -1.055345060;
                }
            }
        } else {
            if (BMI < 23.922422400) {
                if (Age < 58.000000000) {
                    return 0.018644687;
                } else {
                    return 0.646290481;
                }
            } else {
                if (BMI < 24.017253900) {
                    return -0.219110653;
                } else {
                    return -0.006635487;
                }
            }
        }
    }
}

inline double predict_tree_275(double Gender, double Age, double BMI) {
    if (Age < 32.000000000) {
        if (Age < 21.000000000) {
            if (Gender < 1.000000000) {
                return -0.115160279;
            } else {
                return -0.065093413;
            }
        } else {
            if (Age < 22.000000000) {
                if (Gender < 1.000000000) {
                    return 0.004888443;
                } else {
                    return 0.021109061;
                }
            } else {
                if (Age < 31.000000000) {
                    return -0.039450593;
                } else {
                    return -0.086489625;
                }
            }
        }
    } else {
        if (Age < 46.000000000) {
            if (Age < 41.000000000) {
                if (Age < 36.000000000) {
                    return 0.042487170;
                } else {
                    return -0.065200783;
                }
            } else {
                if (Age < 42.000000000) {
                    return 0.174493745;
                } else {
                    return 0.086490609;
                }
            }
        } else {
            if (Age < 47.000000000) {
                if (Gender < 1.000000000) {
                    return -0.235721588;
                } else {
                    return -0.075967379;
                }
            } else {
                if (Age < 61.000000000) {
                    return -0.031506807;
                } else {
                    return 0.009247971;
                }
            }
        }
    }
}

inline double predict_tree_276(double Gender, double Age, double BMI) {
    if (Age < 36.000000000) {
        if (Age < 35.000000000) {
            if (Age < 29.000000000) {
                if (Age < 24.000000000) {
                    return 0.017748045;
                } else {
                    return -0.017544756;
                }
            } else {
                if (Gender < 1.000000000) {
                    return 0.003537388;
                } else {
                    return 0.072169982;
                }
            }
        } else {
            if (Gender < 1.000000000) {
                return 0.012080109;
            } else {
                return 0.244939372;
            }
        }
    } else {
        if (Age < 37.000000000) {
            if (Gender < 1.000000000) {
                return 0.041578587;
            } else {
                return -0.191079855;
            }
        } else {
            if (Age < 48.000000000) {
                if (Age < 47.000000000) {
                    return -0.008317000;
                } else {
                    return 0.280805588;
                }
            } else {
                if (Age < 49.000000000) {
                    return -0.230183095;
                } else {
                    return 0.003314840;
                }
            }
        }
    }
}

inline double predict_tree_277(double Gender, double Age, double BMI) {
    if (BMI < 25.306932400) {
        if (BMI < 24.967113500) {
            if (BMI < 24.930748000) {
                if (Gender < 1.000000000) {
                    return -0.038606729;
                } else {
                    return 0.003112927;
                }
            } else {
                if (Gender < 1.000000000) {
                    return -0.573628604;
                } else {
                    return 0.359534413;
                }
            }
        } else {
            if (BMI < 25.057359700) {
                if (BMI < 24.989587800) {
                    return 0.089038983;
                } else {
                    return 0.396434635;
                }
            } else {
                if (BMI < 25.207756000) {
                    return -0.078694090;
                } else {
                    return 0.143114910;
                }
            }
        }
    } else {
        if (BMI < 25.333334000) {
            return -0.511838675;
        } else {
            if (BMI < 27.173914000) {
                if (BMI < 25.401701000) {
                    return 0.100367442;
                } else {
                    return -0.043876138;
                }
            } else {
                if (BMI < 27.379665400) {
                    return 0.212581590;
                } else {
                    return 0.042384982;
                }
            }
        }
    }
}

inline double predict_tree_278(double Gender, double Age, double BMI) {
    if (BMI < 25.592210800) {
        if (BMI < 25.306932400) {
            if (BMI < 25.249895100) {
                if (BMI < 21.952478400) {
                    return 0.058119375;
                } else {
                    return -0.005924659;
                }
            } else {
                if (BMI < 25.280109400) {
                    return 0.391579092;
                } else {
                    return 0.086970143;
                }
            }
        } else {
            if (BMI < 25.333334000) {
                return -0.576926351;
            } else {
                if (Gender < 1.000000000) {
                    return -0.005586900;
                } else {
                    return -0.325193822;
                }
            }
        }
    } else {
        if (BMI < 25.640243500) {
            if (Gender < 1.000000000) {
                if (BMI < 25.617284800) {
                    return 0.275033057;
                } else {
                    return 0.743257642;
                }
            } else {
                if (BMI < 25.617284800) {
                    return 0.119179808;
                } else {
                    return -0.155189872;
                }
            }
        } else {
            if (BMI < 25.767219500) {
                if (Gender < 1.000000000) {
                    return -0.105370216;
                } else {
                    return 0.334335893;
                }
            } else {
                if (BMI < 25.795917500) {
                    return 0.405047208;
                } else {
                    return 0.036757689;
                }
            }
        }
    }
}

inline double predict_tree_279(double Gender, double Age, double BMI) {
    if (BMI < 26.880950900) {
        if (BMI < 26.528511000) {
            if (BMI < 26.485540400) {
                if (BMI < 24.725183500) {
                    return 0.033110168;
                } else {
                    return -0.018416259;
                }
            } else {
                return -0.547392905;
            }
        } else {
            if (Gender < 1.000000000) {
                if (BMI < 26.570304900) {
                    return 0.255674720;
                } else {
                    return 0.060162019;
                }
            } else {
                if (BMI < 26.794937100) {
                    return 0.268514425;
                } else {
                    return 0.898240507;
                }
            }
        }
    } else {
        if (BMI < 26.979530300) {
            return -0.400085717;
        } else {
            if (BMI < 27.379665400) {
                if (BMI < 27.173914000) {
                    return -0.110542178;
                } else {
                    return 0.174787894;
                }
            } else {
                if (BMI < 27.776929900) {
                    return -0.160145923;
                } else {
                    return -0.084110022;
                }
            }
        }
    }
}

inline double predict_tree_280(double Gender, double Age, double BMI) {
    if (BMI < 21.545091600) {
        if (Age < 47.000000000) {
            if (Age < 45.000000000) {
                if (Age < 37.000000000) {
                    return -0.210506260;
                } else {
                    return 0.168389797;
                }
            } else {
                if (BMI < 21.096191400) {
                    return -0.418272704;
                } else {
                    return -1.531114580;
                }
            }
        } else {
            if (Age < 52.000000000) {
                if (BMI < 20.504934300) {
                    return 0.024213333;
                } else {
                    return 0.980846226;
                }
            } else {
                if (Age < 53.000000000) {
                    return -0.857499838;
                } else {
                    return 0.194110528;
                }
            }
        }
    } else {
        if (BMI < 21.612812000) {
            if (Age < 37.000000000) {
                if (Age < 28.000000000) {
                    return 0.267168164;
                } else {
                    return 1.265480760;
                }
            } else {
                if (Age < 58.000000000) {
                    return -0.533978164;
                } else {
                    return 1.058830140;
                }
            }
        } else {
            if (Age < 42.000000000) {
                if (BMI < 21.989893000) {
                    return 0.210168168;
                } else {
                    return 0.009650497;
                }
            } else {
                if (BMI < 22.862533600) {
                    return -0.137361929;
                } else {
                    return -0.006163036;
                }
            }
        }
    }
}

inline double predict_tree_281(double Gender, double Age, double BMI) {
    if (Age < 23.000000000) {
        if (BMI < 22.282478300) {
            if (BMI < 22.100290300) {
                if (BMI < 21.877550100) {
                    return -0.080898024;
                } else {
                    return 0.482751340;
                }
            } else {
                if (BMI < 22.145328500) {
                    return -1.342564340;
                } else {
                    return -0.613323033;
                }
            }
        } else {
            if (BMI < 22.405876200) {
                if (BMI < 22.347782100) {
                    return 0.671169937;
                } else {
                    return 1.349421380;
                }
            } else {
                if (BMI < 22.991689700) {
                    return -0.205993876;
                } else {
                    return 0.109435201;
                }
            }
        }
    } else {
        if (BMI < 27.173914000) {
            if (BMI < 26.282564200) {
                if (BMI < 26.155626300) {
                    return -0.003776123;
                } else {
                    return 0.269434929;
                }
            } else {
                if (BMI < 26.528511000) {
                    return -0.257472873;
                } else {
                    return 0.012919554;
                }
            }
        } else {
            if (Age < 77.000000000) {
                if (Age < 53.000000000) {
                    return 0.297176570;
                } else {
                    return -0.122413054;
                }
            } else {
                if (BMI < 27.379665400) {
                    return -0.538168311;
                } else {
                    return 0.939665198;
                }
            }
        }
    }
}

inline double predict_tree_282(double Gender, double Age, double BMI) {
    if (BMI < 21.469150500) {
        if (BMI < 21.303949400) {
            if (BMI < 20.796730000) {
                if (BMI < 20.504934300) {
                    return 0.015719522;
                } else {
                    return -0.200653046;
                }
            } else {
                if (BMI < 20.957170500) {
                    return 0.003889763;
                } else {
                    return 0.091817498;
                }
            }
        } else {
            return -0.354681283;
        }
    } else {
        if (BMI < 21.641273500) {
            if (BMI < 21.545091600) {
                return 0.137337163;
            } else {
                if (BMI < 21.612812000) {
                    return 0.272144258;
                } else {
                    return 0.149731055;
                }
            }
        } else {
            if (BMI < 22.230987500) {
                if (BMI < 22.212966900) {
                    return -0.023883605;
                } else {
                    return -0.442938834;
                }
            } else {
                if (BMI < 22.347782100) {
                    return 0.303097188;
                } else {
                    return 0.008153056;
                }
            }
        }
    }
}

inline double predict_tree_283(double Gender, double Age, double BMI) {
    if (BMI < 22.491350200) {
        if (Age < 75.000000000) {
            if (Age < 71.000000000) {
                if (Age < 53.000000000) {
                    return 0.054031704;
                } else {
                    return 0.253337502;
                }
            } else {
                if (BMI < 22.145328500) {
                    return 1.832088710;
                } else {
                    return 0.253979325;
                }
            }
        } else {
            if (BMI < 21.612812000) {
                return 0.612295687;
            } else {
                return -0.907828450;
            }
        }
    } else {
        if (Age < 52.000000000) {
            if (Age < 45.000000000) {
                if (BMI < 22.546575500) {
                    return -0.526490569;
                } else {
                    return 0.014296691;
                }
            } else {
                if (BMI < 23.936061900) {
                    return 0.353047758;
                } else {
                    return 0.064289488;
                }
            }
        } else {
            if (BMI < 25.306932400) {
                if (BMI < 24.989587800) {
                    return -0.035779223;
                } else {
                    return 0.310775191;
                }
            } else {
                if (Age < 53.000000000) {
                    return -0.664226830;
                } else {
                    return -0.072861992;
                }
            }
        }
    }
}

inline double predict_tree_284(double Gender, double Age, double BMI) {
    if (BMI < 24.989587800) {
        if (BMI < 24.976087600) {
            if (BMI < 24.967113500) {
                if (BMI < 24.930748000) {
                    return -0.006690681;
                } else {
                    return -0.269418001;
                }
            } else {
                return 0.336166948;
            }
        } else {
            if (Gender < 1.000000000) {
                return -0.163123488;
            } else {
                return -0.694223404;
            }
        }
    } else {
        if (BMI < 25.057359700) {
            if (BMI < 24.997549100) {
                if (Gender < 1.000000000) {
                    return 0.775743484;
                } else {
                    return -0.136361212;
                }
            } else {
                if (Gender < 1.000000000) {
                    return 0.219258308;
                } else {
                    return 0.582364976;
                }
            }
        } else {
            if (BMI < 25.458065000) {
                if (BMI < 25.436466200) {
                    return 0.004245683;
                } else {
                    return -0.569680631;
                }
            } else {
                if (BMI < 26.423570600) {
                    return 0.081193999;
                } else {
                    return -0.022774601;
                }
            }
        }
    }
}

inline double predict_tree_285(double Gender, double Age, double BMI) {
    if (BMI < 26.869806300) {
        if (BMI < 26.851852400) {
            if (BMI < 20.796730000) {
                return -0.175991282;
            } else {
                if (BMI < 25.992439300) {
                    return 0.012568748;
                } else {
                    return -0.038047083;
                }
            }
        } else {
            return 0.434029847;
        }
    } else {
        if (Gender < 1.000000000) {
            if (BMI < 27.173914000) {
                if (BMI < 27.160493900) {
                    return -0.051482100;
                } else {
                    return -0.403197497;
                }
            } else {
                if (BMI < 27.379665400) {
                    return 0.083653323;
                } else {
                    return -0.060829528;
                }
            }
        } else {
            return 0.367972255;
        }
    }
}

inline double predict_tree_286(double Gender, double Age, double BMI) {
    if (BMI < 22.313278200) {
        if (Age < 23.000000000) {
            if (BMI < 22.060354200) {
                if (BMI < 21.877550100) {
                    return -0.277431697;
                } else {
                    return 0.469996810;
                }
            } else {
                if (BMI < 22.189348200) {
                    return -1.027538780;
                } else {
                    return -0.133919299;
                }
            }
        } else {
            if (Age < 28.000000000) {
                if (BMI < 21.877550100) {
                    return 0.269685745;
                } else {
                    return -0.160767093;
                }
            } else {
                if (Age < 32.000000000) {
                    return -0.251266629;
                } else {
                    return -0.017064484;
                }
            }
        }
    } else {
        if (BMI < 22.347782100) {
            if (Age < 52.000000000) {
                if (Age < 34.000000000) {
                    return 0.222188026;
                } else {
                    return 1.510464430;
                }
            } else {
                return -0.884426653;
            }
        } else {
            if (Age < 79.000000000) {
                if (BMI < 23.147254900) {
                    return 0.059040822;
                } else {
                    return -0.003287996;
                }
            } else {
                if (BMI < 24.997549100) {
                    return -0.153509870;
                } else {
                    return 0.462585390;
                }
            }
        }
    }
}

inline double predict_tree_287(double Gender, double Age, double BMI) {
    if (Age < 61.000000000) {
        if (Age < 42.000000000) {
            if (Age < 41.000000000) {
                if (Age < 21.000000000) {
                    return 0.045018077;
                } else {
                    return -0.002552182;
                }
            } else {
                return 0.123757228;
            }
        } else {
            if (Gender < 1.000000000) {
                if (Age < 44.000000000) {
                    return 0.169593498;
                } else {
                    return -0.027466228;
                }
            } else {
                if (Age < 43.000000000) {
                    return -0.334921956;
                } else {
                    return -0.059460785;
                }
            }
        }
    } else {
        if (Age < 65.000000000) {
            if (Age < 64.000000000) {
                if (Age < 63.000000000) {
                    return 0.156800240;
                } else {
                    return 0.018527396;
                }
            } else {
                if (Gender < 1.000000000) {
                    return 0.147271127;
                } else {
                    return 0.271762073;
                }
            }
        } else {
            if (Age < 79.000000000) {
                if (Age < 75.000000000) {
                    return 0.024401162;
                } else {
                    return 0.070808590;
                }
            } else {
                if (Gender < 1.000000000) {
                    return -0.194550812;
                } else {
                    return 0.098648258;
                }
            }
        }
    }
}

inline double predict_tree_288(double Gender, double Age, double BMI) {
    if (Age < 71.000000000) {
        if (Age < 61.000000000) {
            if (Age < 48.000000000) {
                if (Age < 47.000000000) {
                    return 0.004541038;
                } else {
                    return 0.201587692;
                }
            } else {
                if (BMI < 23.108434700) {
                    return 0.135963574;
                } else {
                    return -0.100522175;
                }
            }
        } else {
            if (BMI < 24.654832800) {
                if (BMI < 24.447279000) {
                    return 0.017329108;
                } else {
                    return -0.565303802;
                }
            } else {
                if (BMI < 25.306932400) {
                    return 0.585073054;
                } else {
                    return 0.036829829;
                }
            }
        }
    } else {
        if (BMI < 22.060354200) {
            if (Age < 79.000000000) {
                return 1.654224870;
            } else {
                if (BMI < 21.303949400) {
                    return 0.658287466;
                } else {
                    return -0.676778555;
                }
            }
        } else {
            if (BMI < 24.323228800) {
                if (BMI < 23.936061900) {
                    return -0.126506373;
                } else {
                    return 0.959165692;
                }
            } else {
                if (BMI < 26.196188000) {
                    return -0.306672990;
                } else {
                    return 0.067494437;
                }
            }
        }
    }
}

inline double predict_tree_289(double Gender, double Age, double BMI) {
    if (Age < 26.000000000) {
        if (Age < 21.000000000) {
            return 0.031243574;
        } else {
            if (Gender < 1.000000000) {
                if (Age < 25.000000000) {
                    return -0.053762626;
                } else {
                    return 0.022498866;
                }
            } else {
                if (Age < 25.000000000) {
                    return -0.057144642;
                } else {
                    return -0.166461691;
                }
            }
        }
    } else {
        if (Age < 27.000000000) {
            if (Gender < 1.000000000) {
                return 0.067287564;
            } else {
                return 0.109204143;
            }
        } else {
            if (Age < 45.000000000) {
                if (Age < 40.000000000) {
                    return 0.009636408;
                } else {
                    return -0.059421666;
                }
            } else {
                if (Gender < 1.000000000) {
                    return 0.051499471;
                } else {
                    return -0.005009268;
                }
            }
        }
    }
}

inline double predict_tree_290(double Gender, double Age, double BMI) {
    if (BMI < 23.437500000) {
        if (Gender < 1.000000000) {
            if (BMI < 22.790329000) {
                if (BMI < 22.720438000) {
                    return -0.231417611;
                } else {
                    return -1.549733760;
                }
            } else {
                if (BMI < 22.829963700) {
                    return 1.482777120;
                } else {
                    return -0.305012614;
                }
            }
        } else {
            if (BMI < 23.374725300) {
                if (BMI < 23.306680700) {
                    return -0.029072704;
                } else {
                    return 0.207059190;
                }
            } else {
                if (BMI < 23.413110700) {
                    return -0.553731322;
                } else {
                    return -0.156809077;
                }
            }
        }
    } else {
        if (BMI < 23.507804900) {
            if (BMI < 23.483476600) {
                if (Gender < 1.000000000) {
                    return 0.183179885;
                } else {
                    return 0.034648128;
                }
            } else {
                return 0.339843094;
            }
        } else {
            if (BMI < 23.597003900) {
                if (BMI < 23.588329300) {
                    return -0.076998457;
                } else {
                    return -0.465330064;
                }
            } else {
                if (BMI < 23.634033200) {
                    return 0.371480286;
                } else {
                    return -0.019842360;
                }
            }
        }
    }
}

inline double predict_tree_291(double Gender, double Age, double BMI) {
    if (Age < 78.000000000) {
        if (Age < 41.000000000) {
            if (Age < 30.000000000) {
                if (Age < 28.000000000) {
                    return -0.018306823;
                } else {
                    return 0.088993147;
                }
            } else {
                if (Age < 31.000000000) {
                    return -0.085703120;
                } else {
                    return -0.037429895;
                }
            }
        } else {
            if (Age < 55.000000000) {
                if (Age < 54.000000000) {
                    return 0.015215092;
                } else {
                    return 0.180716753;
                }
            } else {
                if (Age < 72.000000000) {
                    return -0.034042072;
                } else {
                    return 0.015285715;
                }
            }
        }
    } else {
        if (Gender < 1.000000000) {
            if (Age < 79.000000000) {
                return -0.246558264;
            } else {
                return -0.138515413;
            }
        } else {
            if (Age < 79.000000000) {
                return -0.118090481;
            } else {
                return 0.085193560;
            }
        }
    }
}

inline double predict_tree_292(double Gender, double Age, double BMI) {
    if (Age < 69.000000000) {
        if (Age < 56.000000000) {
            if (Age < 54.000000000) {
                if (Age < 47.000000000) {
                    return -0.008856635;
                } else {
                    return 0.050956678;
                }
            } else {
                if (Gender < 1.000000000) {
                    return 0.141713604;
                } else {
                    return 0.207667440;
                }
            }
        } else {
            if (Age < 63.000000000) {
                if (Age < 62.000000000) {
                    return -0.071486331;
                } else {
                    return -0.300101578;
                }
            } else {
                if (Gender < 1.000000000) {
                    return -0.090185858;
                } else {
                    return 0.091775939;
                }
            }
        }
    } else {
        if (Gender < 1.000000000) {
            if (Age < 73.000000000) {
                if (Age < 70.000000000) {
                    return -0.028386930;
                } else {
                    return 0.279086292;
                }
            } else {
                if (Age < 75.000000000) {
                    return -0.041320287;
                } else {
                    return 0.111560397;
                }
            }
        } else {
            if (Age < 70.000000000) {
                return 0.273382038;
            } else {
                if (Age < 74.000000000) {
                    return -0.139459476;
                } else {
                    return 0.002556593;
                }
            }
        }
    }
}

inline double predict_tree_293(double Gender, double Age, double BMI) {
    if (Age < 36.000000000) {
        if (BMI < 26.312810900) {
            if (BMI < 24.997549100) {
                if (BMI < 24.654832800) {
                    return 0.013253450;
                } else {
                    return 0.303867936;
                }
            } else {
                if (Age < 31.000000000) {
                    return 0.004098727;
                } else {
                    return -0.235865876;
                }
            }
        } else {
            if (Age < 29.000000000) {
                if (BMI < 26.485540400) {
                    return 0.887177169;
                } else {
                    return 0.249794289;
                }
            } else {
                if (BMI < 26.827421200) {
                    return 0.186423853;
                } else {
                    return -0.560059249;
                }
            }
        }
    } else {
        if (BMI < 26.282564200) {
            if (BMI < 26.234567600) {
                if (Age < 37.000000000) {
                    return -0.190023959;
                } else {
                    return 0.010064108;
                }
            } else {
                if (Age < 41.000000000) {
                    return 1.627634410;
                } else {
                    return 0.258279175;
                }
            }
        } else {
            if (Age < 39.000000000) {
                if (Age < 37.000000000) {
                    return 0.143417627;
                } else {
                    return -0.709414721;
                }
            } else {
                if (Age < 51.000000000) {
                    return 0.180058852;
                } else {
                    return -0.109233804;
                }
            }
        }
    }
}

inline double predict_tree_294(double Gender, double Age, double BMI) {
    if (BMI < 22.282478300) {
        if (BMI < 22.189348200) {
            if (BMI < 22.100290300) {
                if (BMI < 22.038566600) {
                    return -0.052834708;
                } else {
                    return -0.250903219;
                }
            } else {
                if (BMI < 22.145328500) {
                    return 0.105691977;
                } else {
                    return 0.173037723;
                }
            }
        } else {
            if (Gender < 1.000000000) {
                return 0.880896866;
            } else {
                if (BMI < 22.230987500) {
                    return -0.346864432;
                } else {
                    return -0.196436077;
                }
            }
        }
    } else {
        if (BMI < 22.476587300) {
            if (Gender < 1.000000000) {
                if (BMI < 22.375679000) {
                    return -1.093676450;
                } else {
                    return 0.390620172;
                }
            } else {
                if (BMI < 22.347782100) {
                    return 0.386370152;
                } else {
                    return 0.166637868;
                }
            }
        } else {
            if (BMI < 22.521507300) {
                if (BMI < 22.491350200) {
                    return -0.235240281;
                } else {
                    return -0.528969824;
                }
            } else {
                if (BMI < 22.582708400) {
                    return 0.423463672;
                } else {
                    return 0.005211976;
                }
            }
        }
    }
}

inline double predict_tree_295(double Gender, double Age, double BMI) {
    if (Age < 31.000000000) {
        if (Gender < 1.000000000) {
            if (Age < 26.000000000) {
                if (Age < 25.000000000) {
                    return -0.029189648;
                } else {
                    return 0.096806288;
                }
            } else {
                if (Age < 28.000000000) {
                    return -0.151869029;
                } else {
                    return -0.058625292;
                }
            }
        } else {
            if (Age < 26.000000000) {
                if (Age < 25.000000000) {
                    return 0.016647207;
                } else {
                    return -0.224022344;
                }
            } else {
                if (Age < 30.000000000) {
                    return 0.071570806;
                } else {
                    return -0.095814839;
                }
            }
        }
    } else {
        if (Age < 38.000000000) {
            if (Age < 37.000000000) {
                if (Age < 36.000000000) {
                    return 0.059399255;
                } else {
                    return -0.115448222;
                }
            } else {
                if (Gender < 1.000000000) {
                    return 0.092436150;
                } else {
                    return 0.320851773;
                }
            }
        } else {
            if (Age < 78.000000000) {
                if (Age < 75.000000000) {
                    return 0.009505859;
                } else {
                    return 0.083890147;
                }
            } else {
                if (Gender < 1.000000000) {
                    return -0.194984600;
                } else {
                    return 0.091146946;
                }
            }
        }
    }
}

inline double predict_tree_296(double Gender, double Age, double BMI) {
    if (BMI < 27.472526600) {
        if (BMI < 21.952478400) {
            if (BMI < 21.913805000) {
                if (BMI < 21.877550100) {
                    return 0.054991115;
                } else {
                    return -0.178913221;
                }
            } else {
                return 0.353630245;
            }
        } else {
            if (BMI < 22.060354200) {
                if (Gender < 1.000000000) {
                    return -0.571173906;
                } else {
                    return -0.265539438;
                }
            } else {
                if (BMI < 22.189348200) {
                    return 0.168715507;
                } else {
                    return 0.003395475;
                }
            }
        }
    } else {
        if (BMI < 27.714790300) {
            return 0.251425594;
        } else {
            if (BMI < 27.776929900) {
                return 0.005914778;
            } else {
                return 0.161682978;
            }
        }
    }
}

inline double predict_tree_297(double Gender, double Age, double BMI) {
    if (Age < 40.000000000) {
        if (Age < 32.000000000) {
            if (Gender < 1.000000000) {
                if (Age < 26.000000000) {
                    return 0.019876745;
                } else {
                    return -0.076055564;
                }
            } else {
                if (Age < 29.000000000) {
                    return 0.008748783;
                } else {
                    return 0.109848887;
                }
            }
        } else {
            if (Age < 36.000000000) {
                if (Age < 35.000000000) {
                    return 0.060744118;
                } else {
                    return 0.131993756;
                }
            } else {
                if (Age < 37.000000000) {
                    return -0.073415950;
                } else {
                    return 0.033307228;
                }
            }
        }
    } else {
        if (Age < 41.000000000) {
            if (Gender < 1.000000000) {
                return -0.372495383;
            } else {
                return -0.042313598;
            }
        } else {
            if (Gender < 1.000000000) {
                if (Age < 44.000000000) {
                    return 0.132854789;
                } else {
                    return 0.006106697;
                }
            } else {
                if (Age < 42.000000000) {
                    return 0.212740719;
                } else {
                    return -0.045352005;
                }
            }
        }
    }
}

inline double predict_tree_298(double Gender, double Age, double BMI) {
    if (Age < 47.000000000) {
        if (Age < 30.000000000) {
            if (Age < 26.000000000) {
                if (Age < 23.000000000) {
                    return -0.001172324;
                } else {
                    return -0.029362000;
                }
            } else {
                if (Gender < 1.000000000) {
                    return -0.034854669;
                } else {
                    return 0.104356341;
                }
            }
        } else {
            if (Age < 38.000000000) {
                if (Age < 36.000000000) {
                    return -0.031044003;
                } else {
                    return -0.137664482;
                }
            } else {
                if (Age < 40.000000000) {
                    return 0.044348273;
                } else {
                    return -0.047400359;
                }
            }
        }
    } else {
        if (Age < 60.000000000) {
            if (Age < 48.000000000) {
                if (Gender < 1.000000000) {
                    return 0.314524740;
                } else {
                    return 0.006527427;
                }
            } else {
                if (Age < 49.000000000) {
                    return -0.129831970;
                } else {
                    return 0.057449613;
                }
            }
        } else {
            if (Age < 61.000000000) {
                if (Gender < 1.000000000) {
                    return -0.072207414;
                } else {
                    return -0.199307114;
                }
            } else {
                if (Age < 62.000000000) {
                    return 0.102635831;
                } else {
                    return -0.024082767;
                }
            }
        }
    }
}

inline double predict_tree_299(double Gender, double Age, double BMI) {
    if (BMI < 25.057359700) {
        if (BMI < 24.989587800) {
            if (BMI < 24.976087600) {
                if (BMI < 24.967113500) {
                    return 0.012343607;
                } else {
                    return 0.368549675;
                }
            } else {
                if (Gender < 1.000000000) {
                    return -0.255186945;
                } else {
                    return -0.497965664;
                }
            }
        } else {
            if (BMI < 24.997549100) {
                return 0.536262631;
            } else {
                if (Gender < 1.000000000) {
                    return 0.125887141;
                } else {
                    return 0.430890948;
                }
            }
        }
    } else {
        if (BMI < 25.082862900) {
            if (Gender < 1.000000000) {
                return -0.502156377;
            } else {
                return -0.255413711;
            }
        } else {
            if (BMI < 25.102392200) {
                if (Gender < 1.000000000) {
                    return 0.567497373;
                } else {
                    return -0.045405913;
                }
            } else {
                if (BMI < 25.469387100) {
                    return -0.091470689;
                } else {
                    return -0.000717104;
                }
            }
        }
    }
}

inline double predict_tree_300(double Gender, double Age, double BMI) {
    if (BMI < 26.851852400) {
        if (BMI < 20.504934300) {
            return 0.277037144;
        } else {
            if (BMI < 26.423570600) {
                if (BMI < 26.312810900) {
                    return -0.012468557;
                } else {
                    return 0.185366824;
                }
            } else {
                if (BMI < 26.528511000) {
                    return -0.300252795;
                } else {
                    return -0.053644460;
                }
            }
        }
    } else {
        if (BMI < 26.869806300) {
            return 0.523406088;
        } else {
            if (BMI < 27.450605400) {
                if (BMI < 27.173914000) {
                    return 0.031804528;
                } else {
                    return 0.279589862;
                }
            } else {
                if (BMI < 27.776929900) {
                    return -0.135008708;
                } else {
                    return 0.001419990;
                }
            }
        }
    }
}

inline double predict_tree_301(double Gender, double Age, double BMI) {
    if (BMI < 24.121749900) {
        if (BMI < 24.088222500) {
            if (Age < 41.000000000) {
                if (BMI < 22.862533600) {
                    return 0.029081807;
                } else {
                    return -0.140176401;
                }
            } else {
                if (Age < 42.000000000) {
                    return 0.439207196;
                } else {
                    return 0.034974124;
                }
            }
        } else {
            if (Age < 72.000000000) {
                if (Age < 44.000000000) {
                    return -0.132088616;
                } else {
                    return -1.091096280;
                }
            } else {
                return 1.245646120;
            }
        }
    } else {
        if (BMI < 24.163265200) {
            if (Age < 69.000000000) {
                if (Age < 55.000000000) {
                    return 0.401627421;
                } else {
                    return -0.670945346;
                }
            } else {
                if (BMI < 24.141519500) {
                    return 0.499803722;
                } else {
                    return 1.910704020;
                }
            }
        } else {
            if (Age < 70.000000000) {
                if (Age < 64.000000000) {
                    return 0.009619511;
                } else {
                    return 0.204199687;
                }
            } else {
                if (Age < 75.000000000) {
                    return -0.211401224;
                } else {
                    return 0.034554139;
                }
            }
        }
    }
}

inline double predict_tree_302(double Gender, double Age, double BMI) {
    if (Age < 42.000000000) {
        if (Age < 22.000000000) {
            if (Age < 21.000000000) {
                if (Gender < 1.000000000) {
                    return -0.011633853;
                } else {
                    return 0.047787637;
                }
            } else {
                if (Gender < 1.000000000) {
                    return -0.037260801;
                } else {
                    return -0.130375877;
                }
            }
        } else {
            if (Age < 24.000000000) {
                if (Gender < 1.000000000) {
                    return 0.086988993;
                } else {
                    return 0.057477515;
                }
            } else {
                if (Age < 37.000000000) {
                    return 0.007041842;
                } else {
                    return 0.050390050;
                }
            }
        }
    } else {
        if (Age < 43.000000000) {
            if (Gender < 1.000000000) {
                return -0.031365525;
            } else {
                return -0.334694296;
            }
        } else {
            if (Age < 66.000000000) {
                if (Age < 64.000000000) {
                    return -0.018695246;
                } else {
                    return 0.120597497;
                }
            } else {
                if (Age < 72.000000000) {
                    return -0.118779838;
                } else {
                    return -0.022203594;
                }
            }
        }
    }
}

inline double predict_tree_303(double Gender, double Age, double BMI) {
    if (BMI < 22.582708400) {
        if (BMI < 22.230987500) {
            if (BMI < 22.212966900) {
                if (Age < 66.000000000) {
                    return -0.008581219;
                } else {
                    return 0.469251364;
                }
            } else {
                if (Age < 46.000000000) {
                    return -0.154595509;
                } else {
                    return -1.228973390;
                }
            }
        } else {
            if (Age < 67.000000000) {
                if (Age < 52.000000000) {
                    return 0.128031418;
                } else {
                    return 0.712642491;
                }
            } else {
                if (BMI < 22.408786800) {
                    return -0.068100013;
                } else {
                    return -0.792212605;
                }
            }
        }
    } else {
        if (BMI < 22.640865300) {
            if (Age < 68.000000000) {
                if (Age < 64.000000000) {
                    return -0.355786592;
                } else {
                    return 1.370101690;
                }
            } else {
                return -1.279199960;
            }
        } else {
            if (BMI < 22.662708300) {
                if (Age < 23.000000000) {
                    return -0.832917511;
                } else {
                    return 0.518446863;
                }
            } else {
                if (BMI < 23.456790900) {
                    return -0.052289914;
                } else {
                    return -0.009841532;
                }
            }
        }
    }
}

inline double predict_tree_304(double Gender, double Age, double BMI) {
    if (BMI < 22.230987500) {
        if (Age < 68.000000000) {
            if (Age < 56.000000000) {
                if (Age < 22.000000000) {
                    return -0.298239201;
                } else {
                    return -0.015854234;
                }
            } else {
                if (BMI < 20.796730000) {
                    return 0.508851707;
                } else {
                    return -0.485135555;
                }
            }
        } else {
            if (Age < 75.000000000) {
                if (BMI < 21.829952200) {
                    return 0.213595316;
                } else {
                    return 1.345912340;
                }
            } else {
                if (BMI < 21.612812000) {
                    return 0.577546656;
                } else {
                    return -0.476313800;
                }
            }
        }
    } else {
        if (BMI < 22.491350200) {
            if (Age < 73.000000000) {
                if (Age < 56.000000000) {
                    return 0.142504409;
                } else {
                    return 0.815607071;
                }
            } else {
                if (BMI < 22.405876200) {
                    return -1.162687180;
                } else {
                    return -0.252858311;
                }
            }
        } else {
            if (BMI < 22.546575500) {
                if (Age < 37.000000000) {
                    return -0.548834085;
                } else {
                    return 0.103324242;
                }
            } else {
                if (BMI < 22.598140700) {
                    return 0.315174043;
                } else {
                    return 0.001710983;
                }
            }
        }
    }
}

inline double predict_tree_305(double Gender, double Age, double BMI) {
    if (BMI < 27.776929900) {
        if (BMI < 26.264944100) {
            if (BMI < 26.078971900) {
                if (BMI < 26.014568300) {
                    return 0.009557438;
                } else {
                    return -0.176736757;
                }
            } else {
                if (BMI < 26.196188000) {
                    return 0.132592455;
                } else {
                    return 0.280021846;
                }
            }
        } else {
            if (BMI < 26.296566000) {
                if (BMI < 26.282564200) {
                    return -0.103937231;
                } else {
                    return -0.615226507;
                }
            } else {
                if (BMI < 26.303619400) {
                    return 0.266879737;
                } else {
                    return -0.064278893;
                }
            }
        }
    } else {
        return 0.384914160;
    }
}

inline double predict_tree_306(double Gender, double Age, double BMI) {
    if (Age < 57.000000000) {
        if (Age < 44.000000000) {
            if (Age < 43.000000000) {
                if (Gender < 1.000000000) {
                    return -0.004518973;
                } else {
                    return 0.019650945;
                }
            } else {
                if (Gender < 1.000000000) {
                    return 0.371862322;
                } else {
                    return -0.111846410;
                }
            }
        } else {
            if (Age < 47.000000000) {
                if (Gender < 1.000000000) {
                    return -0.233218297;
                } else {
                    return 0.059891138;
                }
            } else {
                if (Gender < 1.000000000) {
                    return 0.049894817;
                } else {
                    return -0.071858384;
                }
            }
        }
    } else {
        if (Age < 58.000000000) {
            if (Gender < 1.000000000) {
                return 0.175423652;
            } else {
                return 0.058455948;
            }
        } else {
            if (Age < 59.000000000) {
                if (Gender < 1.000000000) {
                    return -0.223921984;
                } else {
                    return 0.094591461;
                }
            } else {
                if (Gender < 1.000000000) {
                    return 0.064211465;
                } else {
                    return 0.009076617;
                }
            }
        }
    }
}

inline double predict_tree_307(double Gender, double Age, double BMI) {
    if (BMI < 21.367521300) {
        if (Age < 71.000000000) {
            if (Age < 52.000000000) {
                if (Age < 47.000000000) {
                    return 0.156104833;
                } else {
                    return 0.692197323;
                }
            } else {
                if (Age < 55.000000000) {
                    return -1.337429290;
                } else {
                    return -0.033423036;
                }
            }
        } else {
            return 1.084669230;
        }
    } else {
        if (Age < 76.000000000) {
            if (BMI < 21.469150500) {
                if (Age < 26.000000000) {
                    return -0.730336428;
                } else {
                    return -0.088267356;
                }
            } else {
                if (BMI < 27.084159900) {
                    return -0.001996599;
                } else {
                    return -0.101229459;
                }
            }
        } else {
            if (BMI < 25.762678100) {
                if (BMI < 23.456790900) {
                    return -0.494639158;
                } else {
                    return 0.185905814;
                }
            } else {
                if (BMI < 27.084159900) {
                    return -0.720301270;
                } else {
                    return 0.313026667;
                }
            }
        }
    }
}

inline double predict_tree_308(double Gender, double Age, double BMI) {
    if (Age < 37.000000000) {
        if (BMI < 26.851852400) {
            if (BMI < 26.592670400) {
                if (BMI < 25.880571400) {
                    return -0.028340505;
                } else {
                    return 0.088549726;
                }
            } else {
                if (Age < 33.000000000) {
                    return 0.716407061;
                } else {
                    return -0.047147155;
                }
            }
        } else {
            if (Age < 29.000000000) {
                if (BMI < 26.880950900) {
                    return -0.899067163;
                } else {
                    return 0.621407926;
                }
            } else {
                if (Age < 31.000000000) {
                    return -1.370088340;
                } else {
                    return -0.419504762;
                }
            }
        }
    } else {
        if (BMI < 23.808797800) {
            if (Age < 78.000000000) {
                if (Age < 70.000000000) {
                    return 0.061935179;
                } else {
                    return 0.380165726;
                }
            } else {
                if (BMI < 21.612812000) {
                    return 0.564987004;
                } else {
                    return -0.513492942;
                }
            }
        } else {
            if (BMI < 23.849777200) {
                if (Age < 66.000000000) {
                    return -0.454785824;
                } else {
                    return -1.412949800;
                }
            } else {
                if (BMI < 23.875433000) {
                    return 1.060936090;
                } else {
                    return 0.003862098;
                }
            }
        }
    }
}

inline double predict_tree_309(double Gender, double Age, double BMI) {
    if (BMI < 24.569315000) {
        if (Age < 70.000000000) {
            if (Age < 67.000000000) {
                if (BMI < 24.515596400) {
                    return 0.018782906;
                } else {
                    return 0.325992465;
                }
            } else {
                if (BMI < 23.772178600) {
                    return -0.006036607;
                } else {
                    return -0.927270591;
                }
            }
        } else {
            if (BMI < 22.829963700) {
                if (BMI < 22.189348200) {
                    return 0.788183749;
                } else {
                    return -0.544282973;
                }
            } else {
                if (BMI < 23.183391600) {
                    return 0.751775980;
                } else {
                    return 0.243448138;
                }
            }
        }
    } else {
        if (BMI < 24.618103000) {
            if (Age < 71.000000000) {
                if (Age < 68.000000000) {
                    return -0.538952887;
                } else {
                    return 1.503380890;
                }
            } else {
                if (Age < 78.000000000) {
                    return -1.522319320;
                } else {
                    return -0.110411681;
                }
            }
        } else {
            if (Age < 74.000000000) {
                if (Age < 73.000000000) {
                    return 0.010718894;
                } else {
                    return 0.337583274;
                }
            } else {
                if (BMI < 24.956596400) {
                    return -0.901097059;
                } else {
                    return -0.092649013;
                }
            }
        }
    }
}

inline double predict_tree_310(double Gender, double Age, double BMI) {
    if (Age < 77.000000000) {
        if (BMI < 26.869806300) {
            if (BMI < 26.794937100) {
                if (BMI < 26.595745100) {
                    return -0.009586729;
                } else {
                    return -0.395238400;
                }
            } else {
                if (Age < 64.000000000) {
                    return -0.105302028;
                } else {
                    return 1.363160850;
                }
            }
        } else {
            if (Age < 29.000000000) {
                if (BMI < 26.880950900) {
                    return -1.161910180;
                } else {
                    return 0.804075837;
                }
            } else {
                if (Age < 38.000000000) {
                    return -0.657298326;
                } else {
                    return -0.077257291;
                }
            }
        }
    } else {
        if (BMI < 27.084159900) {
            if (BMI < 26.040952700) {
                if (BMI < 25.977846100) {
                    return 0.135141134;
                } else {
                    return 1.214169030;
                }
            } else {
                if (BMI < 26.312810900) {
                    return -2.100065950;
                } else {
                    return -0.164703131;
                }
            }
        } else {
            if (BMI < 27.472526600) {
                if (Age < 78.000000000) {
                    return 0.153879091;
                } else {
                    return 1.202973250;
                }
            } else {
                if (Age < 78.000000000) {
                    return 1.050217270;
                } else {
                    return -0.279239446;
                }
            }
        }
    }
}

inline double predict_tree_311(double Gender, double Age, double BMI) {
    if (BMI < 25.507492100) {
        if (BMI < 25.306932400) {
            if (BMI < 25.227865200) {
                if (BMI < 25.102392200) {
                    return -0.004762601;
                } else {
                    return -0.164077833;
                }
            } else {
                if (BMI < 25.249895100) {
                    return 0.174997270;
                } else {
                    return 0.283286572;
                }
            }
        } else {
            if (BMI < 25.333334000) {
                return -0.481958389;
            } else {
                if (BMI < 25.420015300) {
                    return 0.019883724;
                } else {
                    return -0.204435229;
                }
            }
        }
    } else {
        if (BMI < 26.869806300) {
            if (BMI < 26.851852400) {
                if (BMI < 25.640243500) {
                    return 0.169499427;
                } else {
                    return 0.055736497;
                }
            } else {
                return 0.275595933;
            }
        } else {
            if (BMI < 27.173914000) {
                if (BMI < 27.084159900) {
                    return -0.073954254;
                } else {
                    return -0.374950618;
                }
            } else {
                if (BMI < 27.776929900) {
                    return 0.110537380;
                } else {
                    return -0.277644604;
                }
            }
        }
    }
}

inline double predict_tree_312(double Gender, double Age, double BMI) {
    if (BMI < 23.225431400) {
        if (BMI < 23.183391600) {
            if (Age < 75.000000000) {
                if (Age < 70.000000000) {
                    return -0.028561359;
                } else {
                    return 0.733364940;
                }
            } else {
                if (BMI < 22.829963700) {
                    return -0.803723633;
                } else {
                    return 0.165678367;
                }
            }
        } else {
            if (Age < 36.000000000) {
                if (Age < 33.000000000) {
                    return -0.320517600;
                } else {
                    return 0.320374429;
                }
            } else {
                if (Age < 47.000000000) {
                    return -1.534854890;
                } else {
                    return -0.717233121;
                }
            }
        }
    } else {
        if (BMI < 23.245985000) {
            if (Age < 26.000000000) {
                if (BMI < 23.243408200) {
                    return 0.329830766;
                } else {
                    return 1.091645120;
                }
            } else {
                if (Age < 36.000000000) {
                    return -0.393412530;
                } else {
                    return 0.315945953;
                }
            }
        } else {
            if (BMI < 23.291229200) {
                if (Age < 54.000000000) {
                    return -0.156914502;
                } else {
                    return -0.855182946;
                }
            } else {
                if (BMI < 23.323417700) {
                    return 0.231768340;
                } else {
                    return 0.004384641;
                }
            }
        }
    }
}

inline double predict_tree_313(double Gender, double Age, double BMI) {
    if (Age < 42.000000000) {
        if (Age < 41.000000000) {
            if (Age < 23.000000000) {
                if (Age < 21.000000000) {
                    return -0.040003933;
                } else {
                    return 0.045942321;
                }
            } else {
                if (Age < 34.000000000) {
                    return -0.023162527;
                } else {
                    return 0.007106260;
                }
            }
        } else {
            if (Gender < 1.000000000) {
                return 0.010180530;
            } else {
                return 0.209749788;
            }
        }
    } else {
        if (Age < 52.000000000) {
            if (Age < 51.000000000) {
                if (Age < 49.000000000) {
                    return -0.064008899;
                } else {
                    return 0.007805092;
                }
            } else {
                if (Gender < 1.000000000) {
                    return -0.325492382;
                } else {
                    return 0.014043588;
                }
            }
        } else {
            if (Gender < 1.000000000) {
                if (Age < 55.000000000) {
                    return 0.144230098;
                } else {
                    return -0.002645132;
                }
            } else {
                if (Age < 55.000000000) {
                    return -0.133743957;
                } else {
                    return -0.026333809;
                }
            }
        }
    }
}

inline double predict_tree_314(double Gender, double Age, double BMI) {
    if (Age < 72.000000000) {
        if (BMI < 27.160493900) {
            if (BMI < 27.137414900) {
                if (BMI < 24.212293600) {
                    return -0.017606577;
                } else {
                    return 0.018997792;
                }
            } else {
                if (Age < 66.000000000) {
                    return 0.621227801;
                } else {
                    return -1.163860080;
                }
            }
        } else {
            if (Age < 53.000000000) {
                if (BMI < 27.173914000) {
                    return -0.792711556;
                } else {
                    return 0.255727023;
                }
            } else {
                if (Age < 58.000000000) {
                    return -0.827016413;
                } else {
                    return -0.188859016;
                }
            }
        }
    } else {
        if (BMI < 26.592670400) {
            if (BMI < 26.303619400) {
                if (BMI < 24.323228800) {
                    return 0.192803219;
                } else {
                    return -0.030023867;
                }
            } else {
                if (Age < 79.000000000) {
                    return -0.874893308;
                } else {
                    return 0.364465773;
                }
            }
        } else {
            if (Age < 74.000000000) {
                if (BMI < 26.979530300) {
                    return 2.147911310;
                } else {
                    return 0.007110920;
                }
            } else {
                if (BMI < 27.160493900) {
                    return -0.060345724;
                } else {
                    return 0.463309437;
                }
            }
        }
    }
}

inline double predict_tree_315(double Gender, double Age, double BMI) {
    if (BMI < 23.147254900) {
        if (BMI < 23.051754000) {
            if (Age < 60.000000000) {
                if (Age < 58.000000000) {
                    return -0.008948839;
                } else {
                    return 0.935666561;
                }
            } else {
                if (Age < 65.000000000) {
                    return -0.483938962;
                } else {
                    return 0.031558733;
                }
            }
        } else {
            if (Age < 22.000000000) {
                return -0.581784725;
            } else {
                if (Age < 78.000000000) {
                    return 0.407033980;
                } else {
                    return -1.039972070;
                }
            }
        }
    } else {
        if (BMI < 23.225431400) {
            if (Age < 32.000000000) {
                if (Age < 31.000000000) {
                    return -0.183027744;
                } else {
                    return 0.712107539;
                }
            } else {
                if (Age < 73.000000000) {
                    return -0.785975993;
                } else {
                    return -0.004333191;
                }
            }
        } else {
            if (BMI < 23.243408200) {
                if (Age < 25.000000000) {
                    return -0.416097313;
                } else {
                    return 0.755745828;
                }
            } else {
                if (Age < 21.000000000) {
                    return 0.095311262;
                } else {
                    return -0.023293227;
                }
            }
        }
    }
}

inline double predict_tree_316(double Gender, double Age, double BMI) {
    if (BMI < 24.111507400) {
        if (BMI < 24.088222500) {
            if (BMI < 23.936061900) {
                if (BMI < 23.875433000) {
                    return -0.020180058;
                } else {
                    return 0.245299146;
                }
            } else {
                if (Gender < 1.000000000) {
                    return 0.156613752;
                } else {
                    return -0.246157438;
                }
            }
        } else {
            return -0.639443934;
        }
    } else {
        if (BMI < 24.163265200) {
            if (BMI < 24.141519500) {
                if (Gender < 1.000000000) {
                    return 0.139692441;
                } else {
                    return 0.031349018;
                }
            } else {
                if (BMI < 24.158817300) {
                    return 0.419984877;
                } else {
                    return 0.221413746;
                }
            }
        } else {
            if (BMI < 24.212293600) {
                if (Gender < 1.000000000) {
                    return 0.135502681;
                } else {
                    return -0.552230179;
                }
            } else {
                if (BMI < 24.271844900) {
                    return 0.158470958;
                } else {
                    return -0.000378987;
                }
            }
        }
    }
}

inline double predict_tree_317(double Gender, double Age, double BMI) {
    if (Age < 27.000000000) {
        if (Age < 26.000000000) {
            if (Age < 24.000000000) {
                if (Gender < 1.000000000) {
                    return -0.014199363;
                } else {
                    return 0.044476975;
                }
            } else {
                if (Gender < 1.000000000) {
                    return -0.005922204;
                } else {
                    return -0.061283536;
                }
            }
        } else {
            if (Gender < 1.000000000) {
                return 0.048319634;
            } else {
                return 0.134226963;
            }
        }
    } else {
        if (Age < 37.000000000) {
            if (Age < 32.000000000) {
                if (Age < 31.000000000) {
                    return -0.009887181;
                } else {
                    return 0.028376725;
                }
            } else {
                if (Gender < 1.000000000) {
                    return -0.026521750;
                } else {
                    return -0.089435309;
                }
            }
        } else {
            if (Age < 38.000000000) {
                if (Gender < 1.000000000) {
                    return -0.054106139;
                } else {
                    return 0.342435569;
                }
            } else {
                if (Age < 39.000000000) {
                    return -0.097752675;
                } else {
                    return -0.000218217;
                }
            }
        }
    }
}

inline double predict_tree_318(double Gender, double Age, double BMI) {
    if (BMI < 26.869806300) {
        if (Age < 76.000000000) {
            if (BMI < 26.078971900) {
                if (BMI < 25.992439300) {
                    return 0.022641458;
                } else {
                    return -0.136213630;
                }
            } else {
                if (Age < 71.000000000) {
                    return 0.066124320;
                } else {
                    return 0.577318549;
                }
            }
        } else {
            if (BMI < 25.469387100) {
                if (BMI < 25.458065000) {
                    return -0.008160044;
                } else {
                    return 2.227799890;
                }
            } else {
                if (BMI < 26.332889600) {
                    return -0.658453345;
                } else {
                    return 0.170088723;
                }
            }
        }
    } else {
        if (Age < 72.000000000) {
            if (Age < 71.000000000) {
                if (Age < 29.000000000) {
                    return 0.447085232;
                } else {
                    return -0.153881162;
                }
            } else {
                if (BMI < 26.880950900) {
                    return 0.911785543;
                } else {
                    return -1.768537520;
                }
            }
        } else {
            if (BMI < 27.160493900) {
                if (BMI < 26.880950900) {
                    return 0.630994260;
                } else {
                    return -0.245434836;
                }
            } else {
                if (BMI < 27.450605400) {
                    return 1.104379770;
                } else {
                    return 0.147576883;
                }
            }
        }
    }
}

inline double predict_tree_319(double Gender, double Age, double BMI) {
    if (BMI < 23.795360600) {
        if (Age < 66.000000000) {
            if (BMI < 23.597003900) {
                if (Age < 60.000000000) {
                    return -0.001687721;
                } else {
                    return -0.269054592;
                }
            } else {
                if (Age < 41.000000000) {
                    return 0.013992393;
                } else {
                    return 0.492041051;
                }
            }
        } else {
            if (Age < 75.000000000) {
                if (BMI < 23.291229200) {
                    return 0.184750631;
                } else {
                    return 0.652462006;
                }
            } else {
                if (BMI < 22.829963700) {
                    return -0.531528890;
                } else {
                    return 0.156570166;
                }
            }
        }
    } else {
        if (BMI < 24.654832800) {
            if (Age < 75.000000000) {
                if (Age < 56.000000000) {
                    return -0.037965227;
                } else {
                    return -0.283023596;
                }
            } else {
                if (BMI < 24.023809400) {
                    return -0.201348320;
                } else {
                    return 0.935997605;
                }
            }
        } else {
            if (BMI < 24.725183500) {
                if (Age < 69.000000000) {
                    return 0.346517861;
                } else {
                    return -0.425313830;
                }
            } else {
                if (BMI < 24.772096600) {
                    return -0.286489785;
                } else {
                    return -0.014098391;
                }
            }
        }
    }
}

inline double predict_tree_320(double Gender, double Age, double BMI) {
    if (BMI < 26.332889600) {
        if (BMI < 26.312810900) {
            if (BMI < 26.303619400) {
                if (BMI < 26.155626300) {
                    return 0.009346312;
                } else {
                    return 0.150519699;
                }
            } else {
                return -0.330657393;
            }
        } else {
            if (BMI < 26.315052000) {
                if (Gender < 1.000000000) {
                    return 0.684244394;
                } else {
                    return -0.229447737;
                }
            } else {
                return 0.151125968;
            }
        }
    } else {
        if (BMI < 27.137414900) {
            if (BMI < 26.528511000) {
                if (BMI < 26.423570600) {
                    return -0.067664489;
                } else {
                    return -0.325978696;
                }
            } else {
                if (BMI < 26.570304900) {
                    return 0.120130748;
                } else {
                    return -0.107025541;
                }
            }
        } else {
            if (BMI < 27.160493900) {
                return 0.330727518;
            } else {
                if (BMI < 27.450605400) {
                    return -0.062466383;
                } else {
                    return 0.091786563;
                }
            }
        }
    }
}

inline double predict_tree_321(double Gender, double Age, double BMI) {
    if (Age < 79.000000000) {
        if (Age < 61.000000000) {
            if (Age < 48.000000000) {
                if (Age < 44.000000000) {
                    return 0.003825285;
                } else {
                    return 0.089341164;
                }
            } else {
                if (Age < 60.000000000) {
                    return -0.041074056;
                } else {
                    return -0.228096053;
                }
            }
        } else {
            if (Age < 69.000000000) {
                if (Gender < 1.000000000) {
                    return 0.140284806;
                } else {
                    return 0.056097005;
                }
            } else {
                if (Age < 75.000000000) {
                    return -0.018185805;
                } else {
                    return 0.035859287;
                }
            }
        }
    } else {
        if (Gender < 1.000000000) {
            return -0.337185651;
        } else {
            return -0.068144888;
        }
    }
}

inline double predict_tree_322(double Gender, double Age, double BMI) {
    if (BMI < 21.096191400) {
        if (Age < 51.000000000) {
            if (Age < 47.000000000) {
                if (Age < 46.000000000) {
                    return 0.162339449;
                } else {
                    return -0.716490924;
                }
            } else {
                if (Age < 50.000000000) {
                    return 0.997103870;
                } else {
                    return 0.097645320;
                }
            }
        } else {
            if (Age < 59.000000000) {
                return -1.016458150;
            } else {
                if (BMI < 20.796730000) {
                    return 0.911386907;
                } else {
                    return -0.363018513;
                }
            }
        }
    } else {
        if (BMI < 21.545091600) {
            if (Age < 28.000000000) {
                if (Age < 25.000000000) {
                    return -0.050169129;
                } else {
                    return 0.507767081;
                }
            } else {
                if (Age < 30.000000000) {
                    return -1.150672440;
                } else {
                    return -0.221346647;
                }
            }
        } else {
            if (BMI < 21.641273500) {
                if (Age < 37.000000000) {
                    return 0.640333533;
                } else {
                    return -0.169071868;
                }
            } else {
                if (BMI < 24.023809400) {
                    return -0.016807744;
                } else {
                    return 0.014973161;
                }
            }
        }
    }
}

inline double predict_tree_323(double Gender, double Age, double BMI) {
    if (BMI < 24.956596400) {
        if (BMI < 24.930748000) {
            if (BMI < 23.936061900) {
                if (BMI < 23.849777200) {
                    return 0.000098968;
                } else {
                    return 0.215529397;
                }
            } else {
                if (BMI < 23.939481700) {
                    return -0.651776254;
                } else {
                    return -0.031942163;
                }
            }
        } else {
            if (Gender < 1.000000000) {
                return -0.542220652;
            } else {
                return 0.048211668;
            }
        }
    } else {
        if (BMI < 24.976087600) {
            if (BMI < 24.967113500) {
                if (Gender < 1.000000000) {
                    return -0.097532287;
                } else {
                    return 0.555517256;
                }
            } else {
                if (Gender < 1.000000000) {
                    return 0.584555447;
                } else {
                    return 0.175301269;
                }
            }
        } else {
            if (BMI < 24.989587800) {
                if (Gender < 1.000000000) {
                    return -0.300376981;
                } else {
                    return -0.605472267;
                }
            } else {
                if (BMI < 24.997549100) {
                    return 0.537185311;
                } else {
                    return 0.006882577;
                }
            }
        }
    }
}

inline double predict_tree_324(double Gender, double Age, double BMI) {
    if (BMI < 21.367521300) {
        if (BMI < 20.957170500) {
            if (BMI < 20.796730000) {
                return -0.105037943;
            } else {
                return 0.027133405;
            }
        } else {
            return 0.248718411;
        }
    } else {
        if (BMI < 21.545091600) {
            if (BMI < 21.469150500) {
                return -0.374582767;
            } else {
                return -0.157365054;
            }
        } else {
            if (BMI < 21.641273500) {
                return 0.212331608;
            } else {
                if (BMI < 22.230987500) {
                    return -0.068006374;
                } else {
                    return 0.000173279;
                }
            }
        }
    }
}

inline double predict_tree_325(double Gender, double Age, double BMI) {
    if (Age < 63.000000000) {
        if (Age < 61.000000000) {
            if (Age < 60.000000000) {
                if (Gender < 1.000000000) {
                    return -0.006522158;
                } else {
                    return 0.027891908;
                }
            } else {
                if (Gender < 1.000000000) {
                    return -0.113825910;
                } else {
                    return -0.238941267;
                }
            }
        } else {
            if (Gender < 1.000000000) {
                if (Age < 62.000000000) {
                    return 0.369047076;
                } else {
                    return 0.198822394;
                }
            } else {
                if (Age < 62.000000000) {
                    return -0.165552139;
                } else {
                    return 0.006610553;
                }
            }
        }
    } else {
        if (Age < 78.000000000) {
            if (Age < 75.000000000) {
                if (Age < 74.000000000) {
                    return -0.045102783;
                } else {
                    return -0.147331044;
                }
            } else {
                if (Gender < 1.000000000) {
                    return 0.196276426;
                } else {
                    return -0.091348633;
                }
            }
        } else {
            if (Age < 79.000000000) {
                if (Gender < 1.000000000) {
                    return -0.050990231;
                } else {
                    return -0.236563355;
                }
            } else {
                if (Gender < 1.000000000) {
                    return -0.189534694;
                } else {
                    return 0.016080938;
                }
            }
        }
    }
}

inline double predict_tree_326(double Gender, double Age, double BMI) {
    if (Age < 60.000000000) {
        if (BMI < 27.776929900) {
            if (BMI < 22.230987500) {
                if (Age < 22.000000000) {
                    return -0.301426113;
                } else {
                    return -0.018593961;
                }
            } else {
                if (BMI < 22.347782100) {
                    return 0.333203465;
                } else {
                    return -0.002819256;
                }
            }
        } else {
            if (Age < 57.000000000) {
                if (Age < 53.000000000) {
                    return 0.808838248;
                } else {
                    return -1.375952240;
                }
            } else {
                return 2.144666910;
            }
        }
    } else {
        if (BMI < 22.862533600) {
            if (BMI < 22.837369900) {
                if (BMI < 22.829963700) {
                    return -0.172544554;
                } else {
                    return 1.008255720;
                }
            } else {
                return -1.093586680;
            }
        } else {
            if (BMI < 23.147254900) {
                if (BMI < 23.082542400) {
                    return 0.199478790;
                } else {
                    return 1.109869840;
                }
            } else {
                if (BMI < 23.291229200) {
                    return -0.713180602;
                } else {
                    return -0.034652978;
                }
            }
        }
    }
}

inline double predict_tree_327(double Gender, double Age, double BMI) {
    if (BMI < 21.877550100) {
        if (BMI < 20.796730000) {
            if (BMI < 20.504934300) {
                return -0.152582988;
            } else {
                return -0.020752525;
            }
        } else {
            if (BMI < 21.829952200) {
                if (BMI < 21.612812000) {
                    return 0.149283811;
                } else {
                    return 0.057043966;
                }
            } else {
                return 0.292298675;
            }
        }
    } else {
        if (BMI < 26.851852400) {
            if (BMI < 26.827421200) {
                if (Gender < 1.000000000) {
                    return 0.008664506;
                } else {
                    return -0.035495307;
                }
            } else {
                return -0.424484104;
            }
        } else {
            if (BMI < 26.869806300) {
                return 0.337287545;
            } else {
                if (BMI < 27.472526600) {
                    return 0.055661309;
                } else {
                    return 0.130749941;
                }
            }
        }
    }
}

inline double predict_tree_328(double Gender, double Age, double BMI) {
    if (Age < 75.000000000) {
        if (Age < 73.000000000) {
            if (Age < 55.000000000) {
                if (Age < 54.000000000) {
                    return 0.002467038;
                } else {
                    return 0.235329464;
                }
            } else {
                if (Age < 64.000000000) {
                    return -0.056804087;
                } else {
                    return 0.022909317;
                }
            }
        } else {
            if (Gender < 1.000000000) {
                if (Age < 74.000000000) {
                    return 0.093872465;
                } else {
                    return -0.193622932;
                }
            } else {
                return 0.331902683;
            }
        }
    } else {
        if (Gender < 1.000000000) {
            if (Age < 77.000000000) {
                if (Age < 76.000000000) {
                    return -0.017467292;
                } else {
                    return -0.076733336;
                }
            } else {
                if (Age < 78.000000000) {
                    return 0.116447076;
                } else {
                    return 0.004166041;
                }
            }
        } else {
            if (Age < 79.000000000) {
                if (Age < 76.000000000) {
                    return -0.026777031;
                } else {
                    return -0.194210604;
                }
            } else {
                return 0.026880797;
            }
        }
    }
}

inline double predict_tree_329(double Gender, double Age, double BMI) {
    if (Age < 38.000000000) {
        if (Age < 35.000000000) {
            if (Age < 31.000000000) {
                if (Age < 26.000000000) {
                    return -0.013553664;
                } else {
                    return 0.045441411;
                }
            } else {
                if (Gender < 1.000000000) {
                    return 0.007493300;
                } else {
                    return -0.092806078;
                }
            }
        } else {
            if (Age < 36.000000000) {
                if (Gender < 1.000000000) {
                    return -0.012883183;
                } else {
                    return 0.294728965;
                }
            } else {
                if (Age < 37.000000000) {
                    return 0.003096052;
                } else {
                    return 0.120600261;
                }
            }
        }
    } else {
        if (Age < 43.000000000) {
            if (Age < 42.000000000) {
                if (Age < 39.000000000) {
                    return -0.142424449;
                } else {
                    return -0.038594983;
                }
            } else {
                return -0.187763095;
            }
        } else {
            if (Age < 48.000000000) {
                if (Age < 44.000000000) {
                    return 0.092943206;
                } else {
                    return 0.044022433;
                }
            } else {
                if (Gender < 1.000000000) {
                    return 0.019004053;
                } else {
                    return -0.040395208;
                }
            }
        }
    }
}

inline double predict_tree_330(double Gender, double Age, double BMI) {
    if (Age < 26.000000000) {
        if (BMI < 25.181078000) {
            if (BMI < 24.654832800) {
                if (BMI < 23.733238200) {
                    return 0.001487484;
                } else {
                    return -0.178532407;
                }
            } else {
                if (Age < 25.000000000) {
                    return 0.429459631;
                } else {
                    return -0.501148522;
                }
            }
        } else {
            if (BMI < 25.458065000) {
                if (Age < 24.000000000) {
                    return -0.405346811;
                } else {
                    return -0.856719792;
                }
            } else {
                if (Age < 25.000000000) {
                    return -0.154988021;
                } else {
                    return 0.349384040;
                }
            }
        }
    } else {
        if (Age < 61.000000000) {
            if (BMI < 22.491350200) {
                if (Age < 53.000000000) {
                    return 0.010729534;
                } else {
                    return 0.572287738;
                }
            } else {
                if (BMI < 22.521507300) {
                    return -0.608376622;
                } else {
                    return -0.005187695;
                }
            }
        } else {
            if (BMI < 27.776929900) {
                if (BMI < 25.127830500) {
                    return 0.112389602;
                } else {
                    return -0.005353920;
                }
            } else {
                if (Age < 79.000000000) {
                    return -1.003990890;
                } else {
                    return 1.486304640;
                }
            }
        }
    }
}

inline double predict_tree_331(double Gender, double Age, double BMI) {
    if (BMI < 24.121749900) {
        if (BMI < 24.088222500) {
            if (Age < 75.000000000) {
                if (Age < 70.000000000) {
                    return -0.011750517;
                } else {
                    return 0.224413127;
                }
            } else {
                if (BMI < 22.829963700) {
                    return -0.819683909;
                } else {
                    return 0.041163307;
                }
            }
        } else {
            if (Age < 72.000000000) {
                if (Age < 44.000000000) {
                    return -0.227310076;
                } else {
                    return -0.830126047;
                }
            } else {
                return 1.581877230;
            }
        }
    } else {
        if (BMI < 24.163265200) {
            if (Age < 48.000000000) {
                if (Age < 45.000000000) {
                    return 0.437143445;
                } else {
                    return 1.527910230;
                }
            } else {
                if (Age < 69.000000000) {
                    return -0.298434645;
                } else {
                    return 1.128543620;
                }
            }
        } else {
            if (BMI < 24.212293600) {
                if (Age < 23.000000000) {
                    return 1.168365360;
                } else {
                    return -0.668978691;
                }
            } else {
                if (BMI < 24.271844900) {
                    return 0.277422607;
                } else {
                    return 0.021176159;
                }
            }
        }
    }
}

inline double predict_tree_332(double Gender, double Age, double BMI) {
    if (BMI < 25.992439300) {
        if (BMI < 25.951557200) {
            if (BMI < 25.725517300) {
                if (BMI < 25.469387100) {
                    return 0.008424105;
                } else {
                    return 0.105782211;
                }
            } else {
                if (BMI < 25.737081500) {
                    return -0.434989959;
                } else {
                    return -0.005679397;
                }
            }
        } else {
            if (Gender < 1.000000000) {
                if (BMI < 25.977846100) {
                    return 0.108268544;
                } else {
                    return 0.296105444;
                }
            } else {
                if (BMI < 25.977846100) {
                    return 0.979697406;
                } else {
                    return -0.525748074;
                }
            }
        }
    } else {
        if (BMI < 26.023048400) {
            if (BMI < 26.014568300) {
                return -0.238525212;
            } else {
                return -0.914044142;
            }
        } else {
            if (Gender < 1.000000000) {
                if (BMI < 27.137414900) {
                    return -0.057102397;
                } else {
                    return 0.027933143;
                }
            } else {
                if (BMI < 26.078971900) {
                    return 1.169078590;
                } else {
                    return 0.216126069;
                }
            }
        }
    }
}

inline double predict_tree_333(double Gender, double Age, double BMI) {
    if (BMI < 27.472526600) {
        if (BMI < 27.173914000) {
            if (BMI < 26.282564200) {
                if (BMI < 26.078971900) {
                    return -0.005048085;
                } else {
                    return 0.178919330;
                }
            } else {
                if (Age < 29.000000000) {
                    return 0.184530199;
                } else {
                    return -0.103058666;
                }
            }
        } else {
            if (Age < 75.000000000) {
                if (Age < 29.000000000) {
                    return 1.114995840;
                } else {
                    return -0.061418731;
                }
            } else {
                if (BMI < 27.379665400) {
                    return -0.577514410;
                } else {
                    return 1.622138620;
                }
            }
        }
    } else {
        if (Age < 53.000000000) {
            if (Age < 41.000000000) {
                if (Age < 34.000000000) {
                    return 0.166432142;
                } else {
                    return -1.065104840;
                }
            } else {
                if (Age < 44.000000000) {
                    return 1.482889650;
                } else {
                    return 0.592221797;
                }
            }
        } else {
            if (Age < 79.000000000) {
                if (Age < 78.000000000) {
                    return -0.449363202;
                } else {
                    return -1.298321010;
                }
            } else {
                return 0.471615642;
            }
        }
    }
}

inline double predict_tree_334(double Gender, double Age, double BMI) {
    if (Age < 48.000000000) {
        if (Age < 47.000000000) {
            if (Age < 32.000000000) {
                if (Age < 24.000000000) {
                    return 0.033084061;
                } else {
                    return -0.021299303;
                }
            } else {
                if (Gender < 1.000000000) {
                    return 0.057320878;
                } else {
                    return 0.003821149;
                }
            }
        } else {
            if (Gender < 1.000000000) {
                return 0.270606071;
            } else {
                return 0.050240181;
            }
        }
    } else {
        if (Age < 49.000000000) {
            if (Gender < 1.000000000) {
                return -0.359010190;
            } else {
                return -0.080910824;
            }
        } else {
            if (Age < 58.000000000) {
                if (Age < 57.000000000) {
                    return -0.008624597;
                } else {
                    return -0.184095860;
                }
            } else {
                if (Age < 60.000000000) {
                    return 0.138725698;
                } else {
                    return -0.002011919;
                }
            }
        }
    }
}

inline double predict_tree_335(double Gender, double Age, double BMI) {
    if (BMI < 21.773841900) {
        if (BMI < 21.612812000) {
            if (BMI < 21.367521300) {
                if (BMI < 20.957170500) {
                    return 0.043884758;
                } else {
                    return 0.211618647;
                }
            } else {
                if (BMI < 21.469150500) {
                    return -0.323877782;
                } else {
                    return 0.099385127;
                }
            }
        } else {
            if (BMI < 21.641273500) {
                return 0.379943579;
            } else {
                if (BMI < 21.718065300) {
                    return 0.047132954;
                } else {
                    return 0.232755736;
                }
            }
        }
    } else {
        if (BMI < 22.282478300) {
            if (BMI < 22.189348200) {
                if (BMI < 21.913805000) {
                    return -0.153071493;
                } else {
                    return 0.033333369;
                }
            } else {
                if (Gender < 1.000000000) {
                    return 0.879904985;
                } else {
                    return -0.233241826;
                }
            }
        } else {
            if (BMI < 23.795360600) {
                if (Gender < 1.000000000) {
                    return -0.065693528;
                } else {
                    return 0.072801314;
                }
            } else {
                if (Gender < 1.000000000) {
                    return 0.017832486;
                } else {
                    return -0.042712119;
                }
            }
        }
    }
}

inline double predict_tree_336(double Gender, double Age, double BMI) {
    if (Age < 65.000000000) {
        if (Age < 57.000000000) {
            if (BMI < 25.535446200) {
                if (BMI < 25.436466200) {
                    return -0.020926727;
                } else {
                    return -0.313093722;
                }
            } else {
                if (Age < 51.000000000) {
                    return 0.076469146;
                } else {
                    return -0.130058587;
                }
            }
        } else {
            if (BMI < 23.306680700) {
                if (Age < 60.000000000) {
                    return 0.312888920;
                } else {
                    return -0.500373363;
                }
            } else {
                if (Age < 58.000000000) {
                    return 0.551376879;
                } else {
                    return 0.080243588;
                }
            }
        }
    } else {
        if (BMI < 27.776929900) {
            if (BMI < 25.469387100) {
                if (BMI < 25.458065000) {
                    return 0.002994339;
                } else {
                    return 1.141062500;
                }
            } else {
                if (BMI < 25.510204300) {
                    return -0.737313807;
                } else {
                    return -0.144717187;
                }
            }
        } else {
            if (Age < 67.000000000) {
                return -0.633272231;
            } else {
                if (Age < 78.000000000) {
                    return 1.239056350;
                } else {
                    return -0.178061023;
                }
            }
        }
    }
}

inline double predict_tree_337(double Gender, double Age, double BMI) {
    if (Age < 32.000000000) {
        if (BMI < 25.977846100) {
            if (BMI < 25.854639100) {
                if (BMI < 24.251277900) {
                    return -0.020116560;
                } else {
                    return -0.128619820;
                }
            } else {
                if (Age < 27.000000000) {
                    return -0.501488566;
                } else {
                    return -1.398193960;
                }
            }
        } else {
            if (BMI < 25.992439300) {
                return 1.275159840;
            } else {
                if (Age < 25.000000000) {
                    return -0.114965916;
                } else {
                    return 0.221539855;
                }
            }
        }
    } else {
        if (BMI < 25.992439300) {
            if (BMI < 25.854639100) {
                if (BMI < 25.795917500) {
                    return 0.013626196;
                } else {
                    return -0.371303231;
                }
            } else {
                if (Age < 76.000000000) {
                    return 0.359123707;
                } else {
                    return -1.216700080;
                }
            }
        } else {
            if (BMI < 26.023048400) {
                if (Age < 60.000000000) {
                    return -1.265203120;
                } else {
                    return 0.092651024;
                }
            } else {
                if (BMI < 26.030820800) {
                    return 0.296698213;
                } else {
                    return -0.052445739;
                }
            }
        }
    }
}

inline double predict_tree_338(double Gender, double Age, double BMI) {
    if (BMI < 24.989587800) {
        if (BMI < 24.725183500) {
            if (BMI < 24.654832800) {
                if (BMI < 24.635368300) {
                    return -0.000332191;
                } else {
                    return -0.572287261;
                }
            } else {
                if (BMI < 24.670375800) {
                    return 0.502364337;
                } else {
                    return 0.169147164;
                }
            }
        } else {
            if (BMI < 24.772096600) {
                if (Gender < 1.000000000) {
                    return -0.257956505;
                } else {
                    return -0.455019563;
                }
            } else {
                if (BMI < 24.784257900) {
                    return 0.493431807;
                } else {
                    return -0.099502437;
                }
            }
        }
    } else {
        if (BMI < 25.057359700) {
            if (BMI < 25.029760400) {
                if (Gender < 1.000000000) {
                    return 0.022396151;
                } else {
                    return 0.540346682;
                }
            } else {
                if (Gender < 1.000000000) {
                    return 0.654298544;
                } else {
                    return -0.428697914;
                }
            }
        } else {
            if (Gender < 1.000000000) {
                if (BMI < 25.082862900) {
                    return -0.327855408;
                } else {
                    return 0.035888512;
                }
            } else {
                if (BMI < 26.528511000) {
                    return -0.105070792;
                } else {
                    return 0.628617525;
                }
            }
        }
    }
}

inline double predict_tree_339(double Gender, double Age, double BMI) {
    if (BMI < 23.711845400) {
        if (BMI < 23.665245100) {
            if (Age < 55.000000000) {
                if (BMI < 23.597003900) {
                    return -0.040159740;
                } else {
                    return 0.295207560;
                }
            } else {
                if (Age < 60.000000000) {
                    return 0.340853155;
                } else {
                    return -0.010336015;
                }
            }
        } else {
            if (Age < 58.000000000) {
                if (Age < 22.000000000) {
                    return 0.418455839;
                } else {
                    return -0.847784340;
                }
            } else {
                if (Age < 64.000000000) {
                    return 0.987662613;
                } else {
                    return 0.073186889;
                }
            }
        }
    } else {
        if (BMI < 23.722810700) {
            if (Age < 62.000000000) {
                if (Age < 33.000000000) {
                    return 1.483244540;
                } else {
                    return 0.410135567;
                }
            } else {
                return -1.309571980;
            }
        } else {
            if (Age < 75.000000000) {
                if (Age < 74.000000000) {
                    return 0.019127760;
                } else {
                    return -0.304971814;
                }
            } else {
                if (BMI < 24.355421100) {
                    return 0.882676840;
                } else {
                    return 0.038921915;
                }
            }
        }
    }
}

inline double predict_tree_340(double Gender, double Age, double BMI) {
    if (Age < 61.000000000) {
        if (Age < 59.000000000) {
            if (Age < 23.000000000) {
                if (Gender < 1.000000000) {
                    return 0.035696294;
                } else {
                    return 0.057142027;
                }
            } else {
                if (Age < 57.000000000) {
                    return -0.015533572;
                } else {
                    return 0.039132144;
                }
            }
        } else {
            return 0.127849922;
        }
    } else {
        if (Age < 73.000000000) {
            if (Gender < 1.000000000) {
                if (Age < 72.000000000) {
                    return -0.070410281;
                } else {
                    return 0.262630194;
                }
            } else {
                if (Age < 70.000000000) {
                    return -0.109983824;
                } else {
                    return -0.342285722;
                }
            }
        } else {
            if (Gender < 1.000000000) {
                if (Age < 75.000000000) {
                    return -0.015802844;
                } else {
                    return 0.177963734;
                }
            } else {
                if (Age < 75.000000000) {
                    return 0.223639101;
                } else {
                    return -0.143494099;
                }
            }
        }
    }
}

inline double predict_tree_341(double Gender, double Age, double BMI) {
    if (Age < 73.000000000) {
        if (BMI < 26.570304900) {
            if (BMI < 26.528511000) {
                if (BMI < 23.483476600) {
                    return -0.037250590;
                } else {
                    return 0.006942528;
                }
            } else {
                if (Age < 23.000000000) {
                    return -1.211331610;
                } else {
                    return 0.432848185;
                }
            }
        } else {
            if (Age < 33.000000000) {
                if (Age < 25.000000000) {
                    return -0.414124906;
                } else {
                    return 0.445908546;
                }
            } else {
                if (Age < 39.000000000) {
                    return -0.680544317;
                } else {
                    return -0.046413064;
                }
            }
        }
    } else {
        if (BMI < 26.583175700) {
            if (BMI < 25.617284800) {
                if (BMI < 21.952478400) {
                    return 1.063659310;
                } else {
                    return -0.043209303;
                }
            } else {
                if (Age < 79.000000000) {
                    return -0.675007641;
                } else {
                    return -0.053468361;
                }
            }
        } else {
            if (BMI < 26.880950900) {
                if (Age < 78.000000000) {
                    return 1.193603280;
                } else {
                    return -0.326640308;
                }
            } else {
                if (BMI < 27.160493900) {
                    return -0.415847152;
                } else {
                    return 0.277642429;
                }
            }
        }
    }
}

inline double predict_tree_342(double Gender, double Age, double BMI) {
    if (BMI < 22.230987500) {
        if (Age < 71.000000000) {
            if (Age < 42.000000000) {
                if (Age < 22.000000000) {
                    return -0.210903913;
                } else {
                    return 0.036431957;
                }
            } else {
                if (Age < 53.000000000) {
                    return -0.392915756;
                } else {
                    return 0.025819058;
                }
            }
        } else {
            if (Age < 75.000000000) {
                return 1.505832430;
            } else {
                if (BMI < 21.303949400) {
                    return 0.622226000;
                } else {
                    return -0.540787935;
                }
            }
        }
    } else {
        if (BMI < 22.347782100) {
            if (Age < 46.000000000) {
                if (Age < 38.000000000) {
                    return 0.113365136;
                } else {
                    return 1.936864850;
                }
            } else {
                if (Age < 49.000000000) {
                    return -1.087903260;
                } else {
                    return -0.064672336;
                }
            }
        } else {
            if (BMI < 22.375679000) {
                if (Age < 23.000000000) {
                    return 1.536002520;
                } else {
                    return -1.004152060;
                }
            } else {
                if (BMI < 22.907121700) {
                    return 0.062869288;
                } else {
                    return 0.004338635;
                }
            }
        }
    }
}

inline double predict_tree_343(double Gender, double Age, double BMI) {
    if (BMI < 20.957170500) {
        if (BMI < 20.796730000) {
            if (BMI < 20.504934300) {
                return -0.134788945;
            } else {
                return -0.292996377;
            }
        } else {
            return -0.117146134;
        }
    } else {
        if (BMI < 27.137414900) {
            if (BMI < 27.084159900) {
                if (BMI < 26.312810900) {
                    return -0.003728558;
                } else {
                    return 0.065872721;
                }
            } else {
                return -0.615614712;
            }
        } else {
            if (BMI < 27.714790300) {
                if (BMI < 27.472526600) {
                    return 0.130958527;
                } else {
                    return 0.272338837;
                }
            } else {
                if (BMI < 27.776929900) {
                    return -0.113781080;
                } else {
                    return -0.045357712;
                }
            }
        }
    }
}

inline double predict_tree_344(double Gender, double Age, double BMI) {
    if (BMI < 25.992439300) {
        if (BMI < 25.951557200) {
            if (BMI < 25.795917500) {
                if (BMI < 25.767219500) {
                    return -0.001728419;
                } else {
                    return 0.499079853;
                }
            } else {
                if (Gender < 1.000000000) {
                    return 0.063123472;
                } else {
                    return -0.862224102;
                }
            }
        } else {
            if (BMI < 25.977846100) {
                if (Gender < 1.000000000) {
                    return 0.180592686;
                } else {
                    return 0.326355547;
                }
            } else {
                if (Gender < 1.000000000) {
                    return 0.572677732;
                } else {
                    return -0.522261202;
                }
            }
        }
    } else {
        if (BMI < 26.023048400) {
            if (BMI < 26.014568300) {
                return -0.259516776;
            } else {
                return -0.783439457;
            }
        } else {
            if (BMI < 26.037494700) {
                if (Gender < 1.000000000) {
                    return 0.135411829;
                } else {
                    return 0.620547056;
                }
            } else {
                if (BMI < 26.528511000) {
                    return -0.078430191;
                } else {
                    return 0.005441114;
                }
            }
        }
    }
}

inline double predict_tree_345(double Gender, double Age, double BMI) {
    if (BMI < 26.713068000) {
        if (BMI < 24.989587800) {
            if (BMI < 24.930748000) {
                if (BMI < 24.772096600) {
                    return -0.002643523;
                } else {
                    return 0.103891559;
                }
            } else {
                if (BMI < 24.967113500) {
                    return -0.624722898;
                } else {
                    return -0.055676717;
                }
            }
        } else {
            if (BMI < 24.997549100) {
                if (Gender < 1.000000000) {
                    return 0.806492925;
                } else {
                    return -0.430395156;
                }
            } else {
                if (BMI < 25.102392200) {
                    return 0.158993915;
                } else {
                    return 0.030587230;
                }
            }
        }
    } else {
        if (BMI < 27.472526600) {
            if (BMI < 26.794937100) {
                if (Gender < 1.000000000) {
                    return -0.172374487;
                } else {
                    return -0.786497176;
                }
            } else {
                if (Gender < 1.000000000) {
                    return -0.092409119;
                } else {
                    return 0.991422951;
                }
            }
        } else {
            if (BMI < 27.714790300) {
                return -0.003675560;
            } else {
                if (BMI < 27.776929900) {
                    return 0.205506891;
                } else {
                    return 0.021887004;
                }
            }
        }
    }
}

inline double predict_tree_346(double Gender, double Age, double BMI) {
    if (BMI < 20.504934300) {
        if (Age < 21.000000000) {
            return 1.234533910;
        } else {
            if (Age < 23.000000000) {
                if (Age < 22.000000000) {
                    return -0.808244467;
                } else {
                    return -0.193633318;
                }
            } else {
                if (Age < 26.000000000) {
                    return 0.981146693;
                } else {
                    return -0.073148623;
                }
            }
        }
    } else {
        if (BMI < 22.100290300) {
            if (Age < 28.000000000) {
                if (Age < 27.000000000) {
                    return -0.003715350;
                } else {
                    return 0.487464994;
                }
            } else {
                if (Age < 53.000000000) {
                    return -0.202764004;
                } else {
                    return 0.082372531;
                }
            }
        } else {
            if (BMI < 22.189348200) {
                if (Age < 23.000000000) {
                    return -0.788566172;
                } else {
                    return 0.411650985;
                }
            } else {
                if (BMI < 22.230987500) {
                    return -0.274703413;
                } else {
                    return 0.000391101;
                }
            }
        }
    }
}

inline double predict_tree_347(double Gender, double Age, double BMI) {
    if (BMI < 20.504934300) {
        if (Age < 24.000000000) {
            if (Age < 21.000000000) {
                return 0.808796823;
            } else {
                return -0.621269464;
            }
        } else {
            if (Age < 26.000000000) {
                if (Age < 25.000000000) {
                    return 0.156453848;
                } else {
                    return 1.354061720;
                }
            } else {
                if (Age < 29.000000000) {
                    return -0.274584979;
                } else {
                    return 0.717261493;
                }
            }
        }
    } else {
        if (BMI < 26.851852400) {
            if (Age < 42.000000000) {
                if (BMI < 24.017253900) {
                    return -0.019420823;
                } else {
                    return 0.061491139;
                }
            } else {
                if (BMI < 24.441803000) {
                    return 0.012459192;
                } else {
                    return -0.069156371;
                }
            }
        } else {
            if (Age < 47.000000000) {
                if (Age < 45.000000000) {
                    return -0.028118460;
                } else {
                    return -1.128032680;
                }
            } else {
                if (BMI < 26.869806300) {
                    return 0.876748621;
                } else {
                    return 0.142542213;
                }
            }
        }
    }
}

inline double predict_tree_348(double Gender, double Age, double BMI) {
    if (Age < 64.000000000) {
        if (Age < 63.000000000) {
            if (Age < 61.000000000) {
                if (Gender < 1.000000000) {
                    return -0.007534340;
                } else {
                    return 0.025145719;
                }
            } else {
                if (Gender < 1.000000000) {
                    return 0.283292741;
                } else {
                    return -0.065114468;
                }
            }
        } else {
            if (Gender < 1.000000000) {
                return -0.410732865;
            } else {
                return 0.083124548;
            }
        }
    } else {
        if (Age < 71.000000000) {
            if (Age < 65.000000000) {
                if (Gender < 1.000000000) {
                    return 0.226530164;
                } else {
                    return 0.160659328;
                }
            } else {
                if (Age < 66.000000000) {
                    return -0.073761448;
                } else {
                    return 0.123458520;
                }
            }
        } else {
            if (Gender < 1.000000000) {
                if (Age < 77.000000000) {
                    return 0.019678475;
                } else {
                    return 0.139270499;
                }
            } else {
                if (Age < 73.000000000) {
                    return -0.164026633;
                } else {
                    return -0.041429505;
                }
            }
        }
    }
}

inline double predict_tree_349(double Gender, double Age, double BMI) {
    if (BMI < 26.869806300) {
        if (BMI < 24.956596400) {
            if (BMI < 24.930748000) {
                if (BMI < 24.441803000) {
                    return -0.005780874;
                } else {
                    return -0.043097641;
                }
            } else {
                if (Gender < 1.000000000) {
                    return -0.429451704;
                } else {
                    return 0.068512745;
                }
            }
        } else {
            if (BMI < 25.057359700) {
                if (BMI < 24.989587800) {
                    return 0.078116402;
                } else {
                    return 0.383423060;
                }
            } else {
                if (BMI < 25.082862900) {
                    return -0.201487750;
                } else {
                    return 0.031583264;
                }
            }
        }
    } else {
        if (BMI < 26.880950900) {
            return -0.452388018;
        } else {
            if (BMI < 27.137414900) {
                if (BMI < 27.084159900) {
                    return -0.074529424;
                } else {
                    return -0.302603185;
                }
            } else {
                if (BMI < 27.160493900) {
                    return 0.337014914;
                } else {
                    return -0.066469833;
                }
            }
        }
    }
}

inline double predict_calories(double Gender, double Age, double BMI) {
    double score = 89.016950000; // Base score
    score += predict_tree_0(Gender, Age, BMI);
    score += predict_tree_1(Gender, Age, BMI);
    score += predict_tree_2(Gender, Age, BMI);
    score += predict_tree_3(Gender, Age, BMI);
    score += predict_tree_4(Gender, Age, BMI);
    score += predict_tree_5(Gender, Age, BMI);
    score += predict_tree_6(Gender, Age, BMI);
    score += predict_tree_7(Gender, Age, BMI);
    score += predict_tree_8(Gender, Age, BMI);
    score += predict_tree_9(Gender, Age, BMI);
    score += predict_tree_10(Gender, Age, BMI);
    score += predict_tree_11(Gender, Age, BMI);
    score += predict_tree_12(Gender, Age, BMI);
    score += predict_tree_13(Gender, Age, BMI);
    score += predict_tree_14(Gender, Age, BMI);
    score += predict_tree_15(Gender, Age, BMI);
    score += predict_tree_16(Gender, Age, BMI);
    score += predict_tree_17(Gender, Age, BMI);
    score += predict_tree_18(Gender, Age, BMI);
    score += predict_tree_19(Gender, Age, BMI);
    score += predict_tree_20(Gender, Age, BMI);
    score += predict_tree_21(Gender, Age, BMI);
    score += predict_tree_22(Gender, Age, BMI);
    score += predict_tree_23(Gender, Age, BMI);
    score += predict_tree_24(Gender, Age, BMI);
    score += predict_tree_25(Gender, Age, BMI);
    score += predict_tree_26(Gender, Age, BMI);
    score += predict_tree_27(Gender, Age, BMI);
    score += predict_tree_28(Gender, Age, BMI);
    score += predict_tree_29(Gender, Age, BMI);
    score += predict_tree_30(Gender, Age, BMI);
    score += predict_tree_31(Gender, Age, BMI);
    score += predict_tree_32(Gender, Age, BMI);
    score += predict_tree_33(Gender, Age, BMI);
    score += predict_tree_34(Gender, Age, BMI);
    score += predict_tree_35(Gender, Age, BMI);
    score += predict_tree_36(Gender, Age, BMI);
    score += predict_tree_37(Gender, Age, BMI);
    score += predict_tree_38(Gender, Age, BMI);
    score += predict_tree_39(Gender, Age, BMI);
    score += predict_tree_40(Gender, Age, BMI);
    score += predict_tree_41(Gender, Age, BMI);
    score += predict_tree_42(Gender, Age, BMI);
    score += predict_tree_43(Gender, Age, BMI);
    score += predict_tree_44(Gender, Age, BMI);
    score += predict_tree_45(Gender, Age, BMI);
    score += predict_tree_46(Gender, Age, BMI);
    score += predict_tree_47(Gender, Age, BMI);
    score += predict_tree_48(Gender, Age, BMI);
    score += predict_tree_49(Gender, Age, BMI);
    score += predict_tree_50(Gender, Age, BMI);
    score += predict_tree_51(Gender, Age, BMI);
    score += predict_tree_52(Gender, Age, BMI);
    score += predict_tree_53(Gender, Age, BMI);
    score += predict_tree_54(Gender, Age, BMI);
    score += predict_tree_55(Gender, Age, BMI);
    score += predict_tree_56(Gender, Age, BMI);
    score += predict_tree_57(Gender, Age, BMI);
    score += predict_tree_58(Gender, Age, BMI);
    score += predict_tree_59(Gender, Age, BMI);
    score += predict_tree_60(Gender, Age, BMI);
    score += predict_tree_61(Gender, Age, BMI);
    score += predict_tree_62(Gender, Age, BMI);
    score += predict_tree_63(Gender, Age, BMI);
    score += predict_tree_64(Gender, Age, BMI);
    score += predict_tree_65(Gender, Age, BMI);
    score += predict_tree_66(Gender, Age, BMI);
    score += predict_tree_67(Gender, Age, BMI);
    score += predict_tree_68(Gender, Age, BMI);
    score += predict_tree_69(Gender, Age, BMI);
    score += predict_tree_70(Gender, Age, BMI);
    score += predict_tree_71(Gender, Age, BMI);
    score += predict_tree_72(Gender, Age, BMI);
    score += predict_tree_73(Gender, Age, BMI);
    score += predict_tree_74(Gender, Age, BMI);
    score += predict_tree_75(Gender, Age, BMI);
    score += predict_tree_76(Gender, Age, BMI);
    score += predict_tree_77(Gender, Age, BMI);
    score += predict_tree_78(Gender, Age, BMI);
    score += predict_tree_79(Gender, Age, BMI);
    score += predict_tree_80(Gender, Age, BMI);
    score += predict_tree_81(Gender, Age, BMI);
    score += predict_tree_82(Gender, Age, BMI);
    score += predict_tree_83(Gender, Age, BMI);
    score += predict_tree_84(Gender, Age, BMI);
    score += predict_tree_85(Gender, Age, BMI);
    score += predict_tree_86(Gender, Age, BMI);
    score += predict_tree_87(Gender, Age, BMI);
    score += predict_tree_88(Gender, Age, BMI);
    score += predict_tree_89(Gender, Age, BMI);
    score += predict_tree_90(Gender, Age, BMI);
    score += predict_tree_91(Gender, Age, BMI);
    score += predict_tree_92(Gender, Age, BMI);
    score += predict_tree_93(Gender, Age, BMI);
    score += predict_tree_94(Gender, Age, BMI);
    score += predict_tree_95(Gender, Age, BMI);
    score += predict_tree_96(Gender, Age, BMI);
    score += predict_tree_97(Gender, Age, BMI);
    score += predict_tree_98(Gender, Age, BMI);
    score += predict_tree_99(Gender, Age, BMI);
    score += predict_tree_100(Gender, Age, BMI);
    score += predict_tree_101(Gender, Age, BMI);
    score += predict_tree_102(Gender, Age, BMI);
    score += predict_tree_103(Gender, Age, BMI);
    score += predict_tree_104(Gender, Age, BMI);
    score += predict_tree_105(Gender, Age, BMI);
    score += predict_tree_106(Gender, Age, BMI);
    score += predict_tree_107(Gender, Age, BMI);
    score += predict_tree_108(Gender, Age, BMI);
    score += predict_tree_109(Gender, Age, BMI);
    score += predict_tree_110(Gender, Age, BMI);
    score += predict_tree_111(Gender, Age, BMI);
    score += predict_tree_112(Gender, Age, BMI);
    score += predict_tree_113(Gender, Age, BMI);
    score += predict_tree_114(Gender, Age, BMI);
    score += predict_tree_115(Gender, Age, BMI);
    score += predict_tree_116(Gender, Age, BMI);
    score += predict_tree_117(Gender, Age, BMI);
    score += predict_tree_118(Gender, Age, BMI);
    score += predict_tree_119(Gender, Age, BMI);
    score += predict_tree_120(Gender, Age, BMI);
    score += predict_tree_121(Gender, Age, BMI);
    score += predict_tree_122(Gender, Age, BMI);
    score += predict_tree_123(Gender, Age, BMI);
    score += predict_tree_124(Gender, Age, BMI);
    score += predict_tree_125(Gender, Age, BMI);
    score += predict_tree_126(Gender, Age, BMI);
    score += predict_tree_127(Gender, Age, BMI);
    score += predict_tree_128(Gender, Age, BMI);
    score += predict_tree_129(Gender, Age, BMI);
    score += predict_tree_130(Gender, Age, BMI);
    score += predict_tree_131(Gender, Age, BMI);
    score += predict_tree_132(Gender, Age, BMI);
    score += predict_tree_133(Gender, Age, BMI);
    score += predict_tree_134(Gender, Age, BMI);
    score += predict_tree_135(Gender, Age, BMI);
    score += predict_tree_136(Gender, Age, BMI);
    score += predict_tree_137(Gender, Age, BMI);
    score += predict_tree_138(Gender, Age, BMI);
    score += predict_tree_139(Gender, Age, BMI);
    score += predict_tree_140(Gender, Age, BMI);
    score += predict_tree_141(Gender, Age, BMI);
    score += predict_tree_142(Gender, Age, BMI);
    score += predict_tree_143(Gender, Age, BMI);
    score += predict_tree_144(Gender, Age, BMI);
    score += predict_tree_145(Gender, Age, BMI);
    score += predict_tree_146(Gender, Age, BMI);
    score += predict_tree_147(Gender, Age, BMI);
    score += predict_tree_148(Gender, Age, BMI);
    score += predict_tree_149(Gender, Age, BMI);
    score += predict_tree_150(Gender, Age, BMI);
    score += predict_tree_151(Gender, Age, BMI);
    score += predict_tree_152(Gender, Age, BMI);
    score += predict_tree_153(Gender, Age, BMI);
    score += predict_tree_154(Gender, Age, BMI);
    score += predict_tree_155(Gender, Age, BMI);
    score += predict_tree_156(Gender, Age, BMI);
    score += predict_tree_157(Gender, Age, BMI);
    score += predict_tree_158(Gender, Age, BMI);
    score += predict_tree_159(Gender, Age, BMI);
    score += predict_tree_160(Gender, Age, BMI);
    score += predict_tree_161(Gender, Age, BMI);
    score += predict_tree_162(Gender, Age, BMI);
    score += predict_tree_163(Gender, Age, BMI);
    score += predict_tree_164(Gender, Age, BMI);
    score += predict_tree_165(Gender, Age, BMI);
    score += predict_tree_166(Gender, Age, BMI);
    score += predict_tree_167(Gender, Age, BMI);
    score += predict_tree_168(Gender, Age, BMI);
    score += predict_tree_169(Gender, Age, BMI);
    score += predict_tree_170(Gender, Age, BMI);
    score += predict_tree_171(Gender, Age, BMI);
    score += predict_tree_172(Gender, Age, BMI);
    score += predict_tree_173(Gender, Age, BMI);
    score += predict_tree_174(Gender, Age, BMI);
    score += predict_tree_175(Gender, Age, BMI);
    score += predict_tree_176(Gender, Age, BMI);
    score += predict_tree_177(Gender, Age, BMI);
    score += predict_tree_178(Gender, Age, BMI);
    score += predict_tree_179(Gender, Age, BMI);
    score += predict_tree_180(Gender, Age, BMI);
    score += predict_tree_181(Gender, Age, BMI);
    score += predict_tree_182(Gender, Age, BMI);
    score += predict_tree_183(Gender, Age, BMI);
    score += predict_tree_184(Gender, Age, BMI);
    score += predict_tree_185(Gender, Age, BMI);
    score += predict_tree_186(Gender, Age, BMI);
    score += predict_tree_187(Gender, Age, BMI);
    score += predict_tree_188(Gender, Age, BMI);
    score += predict_tree_189(Gender, Age, BMI);
    score += predict_tree_190(Gender, Age, BMI);
    score += predict_tree_191(Gender, Age, BMI);
    score += predict_tree_192(Gender, Age, BMI);
    score += predict_tree_193(Gender, Age, BMI);
    score += predict_tree_194(Gender, Age, BMI);
    score += predict_tree_195(Gender, Age, BMI);
    score += predict_tree_196(Gender, Age, BMI);
    score += predict_tree_197(Gender, Age, BMI);
    score += predict_tree_198(Gender, Age, BMI);
    score += predict_tree_199(Gender, Age, BMI);
    score += predict_tree_200(Gender, Age, BMI);
    score += predict_tree_201(Gender, Age, BMI);
    score += predict_tree_202(Gender, Age, BMI);
    score += predict_tree_203(Gender, Age, BMI);
    score += predict_tree_204(Gender, Age, BMI);
    score += predict_tree_205(Gender, Age, BMI);
    score += predict_tree_206(Gender, Age, BMI);
    score += predict_tree_207(Gender, Age, BMI);
    score += predict_tree_208(Gender, Age, BMI);
    score += predict_tree_209(Gender, Age, BMI);
    score += predict_tree_210(Gender, Age, BMI);
    score += predict_tree_211(Gender, Age, BMI);
    score += predict_tree_212(Gender, Age, BMI);
    score += predict_tree_213(Gender, Age, BMI);
    score += predict_tree_214(Gender, Age, BMI);
    score += predict_tree_215(Gender, Age, BMI);
    score += predict_tree_216(Gender, Age, BMI);
    score += predict_tree_217(Gender, Age, BMI);
    score += predict_tree_218(Gender, Age, BMI);
    score += predict_tree_219(Gender, Age, BMI);
    score += predict_tree_220(Gender, Age, BMI);
    score += predict_tree_221(Gender, Age, BMI);
    score += predict_tree_222(Gender, Age, BMI);
    score += predict_tree_223(Gender, Age, BMI);
    score += predict_tree_224(Gender, Age, BMI);
    score += predict_tree_225(Gender, Age, BMI);
    score += predict_tree_226(Gender, Age, BMI);
    score += predict_tree_227(Gender, Age, BMI);
    score += predict_tree_228(Gender, Age, BMI);
    score += predict_tree_229(Gender, Age, BMI);
    score += predict_tree_230(Gender, Age, BMI);
    score += predict_tree_231(Gender, Age, BMI);
    score += predict_tree_232(Gender, Age, BMI);
    score += predict_tree_233(Gender, Age, BMI);
    score += predict_tree_234(Gender, Age, BMI);
    score += predict_tree_235(Gender, Age, BMI);
    score += predict_tree_236(Gender, Age, BMI);
    score += predict_tree_237(Gender, Age, BMI);
    score += predict_tree_238(Gender, Age, BMI);
    score += predict_tree_239(Gender, Age, BMI);
    score += predict_tree_240(Gender, Age, BMI);
    score += predict_tree_241(Gender, Age, BMI);
    score += predict_tree_242(Gender, Age, BMI);
    score += predict_tree_243(Gender, Age, BMI);
    score += predict_tree_244(Gender, Age, BMI);
    score += predict_tree_245(Gender, Age, BMI);
    score += predict_tree_246(Gender, Age, BMI);
    score += predict_tree_247(Gender, Age, BMI);
    score += predict_tree_248(Gender, Age, BMI);
    score += predict_tree_249(Gender, Age, BMI);
    score += predict_tree_250(Gender, Age, BMI);
    score += predict_tree_251(Gender, Age, BMI);
    score += predict_tree_252(Gender, Age, BMI);
    score += predict_tree_253(Gender, Age, BMI);
    score += predict_tree_254(Gender, Age, BMI);
    score += predict_tree_255(Gender, Age, BMI);
    score += predict_tree_256(Gender, Age, BMI);
    score += predict_tree_257(Gender, Age, BMI);
    score += predict_tree_258(Gender, Age, BMI);
    score += predict_tree_259(Gender, Age, BMI);
    score += predict_tree_260(Gender, Age, BMI);
    score += predict_tree_261(Gender, Age, BMI);
    score += predict_tree_262(Gender, Age, BMI);
    score += predict_tree_263(Gender, Age, BMI);
    score += predict_tree_264(Gender, Age, BMI);
    score += predict_tree_265(Gender, Age, BMI);
    score += predict_tree_266(Gender, Age, BMI);
    score += predict_tree_267(Gender, Age, BMI);
    score += predict_tree_268(Gender, Age, BMI);
    score += predict_tree_269(Gender, Age, BMI);
    score += predict_tree_270(Gender, Age, BMI);
    score += predict_tree_271(Gender, Age, BMI);
    score += predict_tree_272(Gender, Age, BMI);
    score += predict_tree_273(Gender, Age, BMI);
    score += predict_tree_274(Gender, Age, BMI);
    score += predict_tree_275(Gender, Age, BMI);
    score += predict_tree_276(Gender, Age, BMI);
    score += predict_tree_277(Gender, Age, BMI);
    score += predict_tree_278(Gender, Age, BMI);
    score += predict_tree_279(Gender, Age, BMI);
    score += predict_tree_280(Gender, Age, BMI);
    score += predict_tree_281(Gender, Age, BMI);
    score += predict_tree_282(Gender, Age, BMI);
    score += predict_tree_283(Gender, Age, BMI);
    score += predict_tree_284(Gender, Age, BMI);
    score += predict_tree_285(Gender, Age, BMI);
    score += predict_tree_286(Gender, Age, BMI);
    score += predict_tree_287(Gender, Age, BMI);
    score += predict_tree_288(Gender, Age, BMI);
    score += predict_tree_289(Gender, Age, BMI);
    score += predict_tree_290(Gender, Age, BMI);
    score += predict_tree_291(Gender, Age, BMI);
    score += predict_tree_292(Gender, Age, BMI);
    score += predict_tree_293(Gender, Age, BMI);
    score += predict_tree_294(Gender, Age, BMI);
    score += predict_tree_295(Gender, Age, BMI);
    score += predict_tree_296(Gender, Age, BMI);
    score += predict_tree_297(Gender, Age, BMI);
    score += predict_tree_298(Gender, Age, BMI);
    score += predict_tree_299(Gender, Age, BMI);
    score += predict_tree_300(Gender, Age, BMI);
    score += predict_tree_301(Gender, Age, BMI);
    score += predict_tree_302(Gender, Age, BMI);
    score += predict_tree_303(Gender, Age, BMI);
    score += predict_tree_304(Gender, Age, BMI);
    score += predict_tree_305(Gender, Age, BMI);
    score += predict_tree_306(Gender, Age, BMI);
    score += predict_tree_307(Gender, Age, BMI);
    score += predict_tree_308(Gender, Age, BMI);
    score += predict_tree_309(Gender, Age, BMI);
    score += predict_tree_310(Gender, Age, BMI);
    score += predict_tree_311(Gender, Age, BMI);
    score += predict_tree_312(Gender, Age, BMI);
    score += predict_tree_313(Gender, Age, BMI);
    score += predict_tree_314(Gender, Age, BMI);
    score += predict_tree_315(Gender, Age, BMI);
    score += predict_tree_316(Gender, Age, BMI);
    score += predict_tree_317(Gender, Age, BMI);
    score += predict_tree_318(Gender, Age, BMI);
    score += predict_tree_319(Gender, Age, BMI);
    score += predict_tree_320(Gender, Age, BMI);
    score += predict_tree_321(Gender, Age, BMI);
    score += predict_tree_322(Gender, Age, BMI);
    score += predict_tree_323(Gender, Age, BMI);
    score += predict_tree_324(Gender, Age, BMI);
    score += predict_tree_325(Gender, Age, BMI);
    score += predict_tree_326(Gender, Age, BMI);
    score += predict_tree_327(Gender, Age, BMI);
    score += predict_tree_328(Gender, Age, BMI);
    score += predict_tree_329(Gender, Age, BMI);
    score += predict_tree_330(Gender, Age, BMI);
    score += predict_tree_331(Gender, Age, BMI);
    score += predict_tree_332(Gender, Age, BMI);
    score += predict_tree_333(Gender, Age, BMI);
    score += predict_tree_334(Gender, Age, BMI);
    score += predict_tree_335(Gender, Age, BMI);
    score += predict_tree_336(Gender, Age, BMI);
    score += predict_tree_337(Gender, Age, BMI);
    score += predict_tree_338(Gender, Age, BMI);
    score += predict_tree_339(Gender, Age, BMI);
    score += predict_tree_340(Gender, Age, BMI);
    score += predict_tree_341(Gender, Age, BMI);
    score += predict_tree_342(Gender, Age, BMI);
    score += predict_tree_343(Gender, Age, BMI);
    score += predict_tree_344(Gender, Age, BMI);
    score += predict_tree_345(Gender, Age, BMI);
    score += predict_tree_346(Gender, Age, BMI);
    score += predict_tree_347(Gender, Age, BMI);
    score += predict_tree_348(Gender, Age, BMI);
    score += predict_tree_349(Gender, Age, BMI);
    return score;
}


// Predict Stroke Risk Function
inline double sigmoid(double z) {
    return 1.0 / (1.0 + std::exp(-z));
}

inline double predict_stroke_probability(
    double age, double hypertension, double heart_disease, double avg_glucose_level, double bmi,
    double avg_RestingBP, double heart_cholesterol, double avg_MaxHR, double avg_Oldpeak,
    double heart_disease_rate, double estimated_calories
) {
    double features[11] = {
        age, hypertension, heart_disease, avg_glucose_level, bmi,
        avg_RestingBP, heart_cholesterol, avg_MaxHR, avg_Oldpeak,
        heart_disease_rate, estimated_calories
    };

    double z = LR_INTERCEPT;
    for (int i = 0; i < 11; ++i) {
        double scaled_val = (features[i] - SCALER_MEAN[i]) / SCALER_SCALE[i];
        z += LR_COEFFICIENTS[i] * scaled_val;
    }

    return sigmoid(z);
}

} // namespace StrokeModel

#endif // STROKE_PREDICTOR_MODEL_H
