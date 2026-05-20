#include <iostream>
#include <string>
#include <sstream>
#include <iomanip>
#include "stroke_predictor_model.h"

// Helper function to safely read double input with fallback
double get_double_input(const std::string& prompt, double fallback) {
    std::cout << prompt << " [Default: " << fallback << "]: ";
    std::string input;
    std::getline(std::cin, input);
    if (input.empty()) {
        return fallback;
    }
    std::stringstream ss(input);
    double val;
    if (ss >> val) {
        return val;
    }
    std::cout << "-> Invalid input. Using default: " << fallback << "\n";
    return fallback;
}

// Helper to safely read binary input (0 or 1) with fallback
double get_binary_input(const std::string& prompt, double fallback) {
    while (true) {
        std::cout << prompt << " (0=No, 1=Yes) [Default: " << (int)fallback << "]: ";
        std::string input;
        std::getline(std::cin, input);
        if (input.empty()) {
            return fallback;
        }
        std::stringstream ss(input);
        int val;
        if (ss >> val && (val == 0 || val == 1)) {
            return static_cast<double>(val);
        }
        std::cout << "-> Invalid selection. Please enter 0 or 1.\n";
    }
}

// Helper to safely read gender with fallback
double get_gender_input(double fallback) {
    while (true) {
        std::cout << "Gender (0=Male, 1=Female) [Default: " << (fallback == 1.0 ? "Female" : "Male") << "]: ";
        std::string input;
        std::getline(std::cin, input);
        if (input.empty()) {
            return fallback;
        }
        std::stringstream ss(input);
        int val;
        if (ss >> val && (val == 0 || val == 1)) {
            return static_cast<double>(val);
        }
        std::cout << "-> Invalid selection. Please enter 0 for Male or 1 for Female.\n";
    }
}

int main() {
    std::cout << "========================================================\n";
    std::cout << "        STROKE RISK PREDICTOR C++ INTERFACE             \n";
    std::cout << "========================================================\n";
    std::cout << "Please enter patient details below (Press Enter to use default).\n\n";

    // Read inputs using medians from metadata as defaults
    double gender = get_gender_input(StrokeModel::MEDIANS.gender);
    double age = get_double_input("Age (Years)", StrokeModel::MEDIANS.age);
    double hypertension = get_binary_input("Hypertension history", StrokeModel::MEDIANS.hypertension);
    double heart_disease = get_binary_input("Heart disease history", StrokeModel::MEDIANS.heart_disease);
    double avg_glucose_level = get_double_input("Average glucose level (mg/dL)", StrokeModel::MEDIANS.avg_glucose_level);
    
    // Clean and validate BMI bounds like in preprocessor
    double bmi = get_double_input("Body Mass Index (BMI)", StrokeModel::MEDIANS.bmi);
    if (bmi < 10.0 || bmi > 80.0) {
        std::cout << "-> BMI is out of clinical bounds (10-80). Auto-correcting to median: " << StrokeModel::MEDIANS.bmi << "\n";
        bmi = StrokeModel::MEDIANS.bmi;
    }

    double avg_RestingBP = get_double_input("Resting Blood Pressure (mmHg)", StrokeModel::MEDIANS.avg_RestingBP);
    double heart_cholesterol = get_double_input("Cholesterol level (mg/dL)", StrokeModel::MEDIANS.heart_cholesterol);
    double avg_MaxHR = get_double_input("Maximum Heart Rate", StrokeModel::MEDIANS.avg_MaxHR);
    double avg_Oldpeak = get_double_input("ST depression (Oldpeak)", StrokeModel::MEDIANS.avg_Oldpeak);
    double heart_disease_rate = get_double_input("Heart disease rate", StrokeModel::MEDIANS.heart_disease_rate);

    std::cout << "\n--------------------------------------------------------\n";
    std::cout << "Processing models...\n";

    // 1. Estimate Calories using transpiled XGBoost Regressor
    double est_calories = StrokeModel::predict_calories(gender, age, bmi);
    std::cout << "Predicted Calories burned: " << std::fixed << std::setprecision(2) << est_calories << " kcal\n";

    // 2. Predict Stroke Probability using scaled Logistic Regression
    double prob = StrokeModel::predict_stroke_probability(
        age, hypertension, heart_disease, avg_glucose_level, bmi,
        avg_RestingBP, heart_cholesterol, avg_MaxHR, avg_Oldpeak,
        heart_disease_rate, est_calories
    );

    bool prediction = (prob >= StrokeModel::DECISION_THRESHOLD);

    std::cout << "\n=================== RESULTS ============================\n";
    std::cout << "Stroke Risk Probability: " << std::fixed << std::setprecision(4) << (prob * 100.0) << "%\n";
    std::cout << "Classification Threshold: " << (StrokeModel::DECISION_THRESHOLD * 100.0) << "%\n";
    std::cout << "Stroke Prediction Class: " << (prediction ? "POSITIVE (High Risk)" : "NEGATIVE (Low Risk)") << "\n";
    std::cout << "========================================================\n";

    std::cout << "\nPress Enter to exit.";
    std::string exit_input;
    std::getline(std::cin, exit_input);

    return 0;
}
