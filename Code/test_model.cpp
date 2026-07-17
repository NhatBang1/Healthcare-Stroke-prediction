#include <iostream>
#include <iomanip>
#include "../artifacts/stroke_predictor_model.h"

int main() {
    std::cout << "=========================================================\n";
    std::cout << "   STROKE RISK PREDICTION - C++ EDGE INFERENCE TESTER\n";
    std::cout << "=========================================================\n\n";

    std::cout << "Model Decision Threshold: " << StrokeModel::DECISION_THRESHOLD << "\n\n";

    // Test Case 1: High risk patient
    // Age: 67, Male (Gender=0), Hypertension: 1, Heart Disease: 1, Glucose: 228.69, BMI: 36.6
    // RestingBP: 140, Cholesterol: 245, MaxHR: 130, Oldpeak: 1.5, Heart Disease Rate: 0.8
    double gender_1 = 0.0; // Male
    double age_1 = 67.0;
    double hypertension_1 = 1.0;
    double heart_disease_1 = 1.0;
    double glucose_1 = 228.69;
    double bmi_1 = 36.6;
    double resting_bp_1 = 140.0;
    double cholesterol_1 = 245.0;
    double max_hr_1 = 130.0;
    double oldpeak_1 = 1.5;
    double heart_disease_rate_1 = 0.8;

    // 1. Synthesize calories
    double calories_1 = StrokeModel::predict_calories(gender_1, age_1, bmi_1);

    // 2. Predict stroke probability
    double prob_1 = StrokeModel::predict_stroke_probability(
        age_1, hypertension_1, heart_disease_1, glucose_1, bmi_1,
        resting_bp_1, cholesterol_1, max_hr_1, oldpeak_1,
        heart_disease_rate_1, calories_1
    );

    std::cout << "--- TEST PATIENT 1 (High Risk Clinical Profile) ---\n";
    std::cout << "Gender: Male, Age: " << age_1 << ", BMI: " << bmi_1 << "\n";
    std::cout << "Synthesized Estimated_calories: " << std::fixed << std::setprecision(4) << calories_1 << " kcal\n";
    std::cout << "Predicted Stroke Probability:   " << prob_1 * 100.0 << "%\n";
    std::cout << "Decision Outcome:               " 
              << (prob_1 >= StrokeModel::DECISION_THRESHOLD ? "🔴 STROKE RISK DETECTED (High Risk)" : "🟢 NORMAL (Low Risk)") 
              << "\n\n";

    // Test Case 2: Low risk patient
    // Age: 35, Female (Gender=1), Hypertension: 0, Heart Disease: 0, Glucose: 85.3, BMI: 22.1
    // RestingBP: 120, Cholesterol: 180, MaxHR: 165, Oldpeak: 0.0, Heart Disease Rate: 0.1
    double gender_2 = 1.0; // Female
    double age_2 = 35.0;
    double hypertension_2 = 0.0;
    double heart_disease_2 = 0.0;
    double glucose_2 = 85.3;
    double bmi_2 = 22.1;
    double resting_bp_2 = 120.0;
    double cholesterol_2 = 180.0;
    double max_hr_2 = 165.0;
    double oldpeak_2 = 0.0;
    double heart_disease_rate_2 = 0.1;

    double calories_2 = StrokeModel::predict_calories(gender_2, age_2, bmi_2);
    double prob_2 = StrokeModel::predict_stroke_probability(
        age_2, hypertension_2, heart_disease_2, glucose_2, bmi_2,
        resting_bp_2, cholesterol_2, max_hr_2, oldpeak_2,
        heart_disease_rate_2, calories_2
    );

    std::cout << "--- TEST PATIENT 2 (Low Risk Clinical Profile) ---\n";
    std::cout << "Gender: Female, Age: " << age_2 << ", BMI: " << bmi_2 << "\n";
    std::cout << "Synthesized Estimated_calories: " << calories_2 << " kcal\n";
    std::cout << "Predicted Stroke Probability:   " << prob_2 * 100.0 << "%\n";
    std::cout << "Decision Outcome:               " 
              << (prob_2 >= StrokeModel::DECISION_THRESHOLD ? "🔴 STROKE RISK DETECTED (High Risk)" : "🟢 NORMAL (Low Risk)") 
              << "\n\n";

    std::cout << "=========================================================\n";
    return 0;
}
