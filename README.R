# ==============================================================================
# PROJECT: BioSentinel Space Radiation & Astrobiology Predictive AI Platform
# AUTHOR: Academic Scholar (Johns Hopkins University)
# TARGET: Handshake AI Portfolio Showcase
# ==============================================================================

# ------------------------------------------------------------------------------
# STEP 1: Environment Setup & Package Management
# ------------------------------------------------------------------------------
required_packages <- c("tidyverse", "caret", "randomForest", "ggplot2")
missing_packages <- required_packages[!(required_packages %in% installed.packages()[,"Package"])]

if(length(missing_packages)) {
  print(paste("Installing missing dependencies:", paste(missing_packages, collapse = ", ")))
  install.packages(missing_packages, repos = "https://cloud.r-project.org")
}

library(tidyverse)
library(caret)
library(randomForest)
library(ggplot2)

# Create structural directories programmatically if they don't exist
if(!dir.exists("data")) dir.create("data")
if(!dir.exists("models")) dir.create("models")
if(!dir.exists("output")) dir.create("output")

# Set seed for reproducible results (Crucial for recruiter technical verification)
set.seed(42)

# ------------------------------------------------------------------------------
# STEP 2: Synthetic Telemetry Generation & Feature Engineering
# ------------------------------------------------------------------------------
print("Generating BioSentinel deep space telemetry data streams...")
n_observations <- 1200

telemetry_data <- tibble(
  timestamp = seq(as.POSIXct("2026-01-01 00:00:00"), by = "hour", length.out = n_observations),
  LET_index = runif(n_observations, min = 0.5, max = 15.0),    # Linear Energy Transfer (Radiation quality)
  TID_krad  = runif(n_observations, min = 0.1, max = 5.0),     # Total Ionizing Dose accumulation
  solar_flare_active = if_else(LET_index > 11.5 | TID_krad > 4.0, 1, 0),
  payload_temp_c = rnorm(n_observations, mean = 28, sd = 0.6)  # Internal incubator thermal control
) %>%
  # Simulate biological cell mutations/survival based on actual astrobiology constraints
  mutate(
    # Wild Type strain handles DNA repair loops moderately well
    WT_survival_rate = 100 - (0.7 * LET_index) - (2.0 * TID_krad) + rnorm(n_observations, 0, 1.5),
    # Rad51 mutant strain completely lacks double-strand break repair capabilities
    rad51_survival_rate = 100 - (1.8 * LET_index) - (4.2 * TID_krad) - (5.0 * solar_flare_active) + rnorm(n_observations, 0, 2.5)
  ) %>%
  # Constrain data within logical bounds (0% to 100% viability)
  mutate(
    WT_survival_rate = pmax(pmin(WT_survival_rate, 100), 0),
    rad51_survival_rate = pmax(pmin(rad51_survival_rate, 100), 0)
  )

# Export processed artifact to data directory
write_csv(telemetry_data, "data/biosentinel_processed_telemetry.csv")
print("Data pipeline step completed. Dataset saved to 'data/biosentinel_processed_telemetry.csv'")

# ------------------------------------------------------------------------------
# STEP 3: Machine Learning Engine Preparation
# ------------------------------------------------------------------------------
print("Preparing dataset partition matrices...")

# Focus task: Predict biological degradation of the high-risk rad51 deletion strain
ml_data <- telemetry_data %>%
  select(LET_index, TID_krad, solar_flare_active, payload_temp_c, rad51_survival_rate)

# Split data: 80% Training for model fitting, 20% Testing for validation评估
train_index <- createDataPartition(ml_data$rad51_survival_rate, p = 0.8, list = FALSE)
train_set   <- ml_data[train_index, ]
test_set    <- ml_data[-train_index, ]

# ------------------------------------------------------------------------------
# STEP 4: AI Model Architecture Execution
# ------------------------------------------------------------------------------
print("Training Random Forest Ensemble Core...")
ai_core <- randomForest(
  rad51_survival_rate ~ ., 
  data = train_set, 
  ntree = 200, 
  importance = TRUE
)

print("AI Training Phase Completed.")

# ------------------------------------------------------------------------------
# STEP 5: Validation, Metrics Collection & Model Serialization
# ------------------------------------------------------------------------------
predictions <- predict(ai_core, test_set)
rmse_score  <- RMSE(predictions, test_set$rad51_survival_rate)
r2_score    <- R2(predictions, test_set$rad51_survival_rate)

print("============ ML MODEL PERFORMANCE EVALUATION ============")
print(paste("Root Mean Squared Error (RMSE):", round(rmse_score, 4)))
print(paste("Coefficient of Determination (R²):", round(r2_score, 4)))
print("=========================================================")

# Save model binary file to disk
saveRDS(ai_core, "models/biosentinel_rf_model.rds")

# ------------------------------------------------------------------------------
# STEP 6: Feature Importance Data Visualization
# ------------------------------------------------------------------------------
importance_df <- as.data.frame(importance(ai_core))
importance_df$Feature <- rownames(importance_df)

ggplot(importance_df, aes(x = reorder(Feature, IncNodePurity), y = IncNodePurity)) +
  geom_bar(stat = "identity", fill = "#2c3e50", width = 0.6) +
  coord_flip() +
  labs(
    title = "BioSentinel AI Engine: Feature Importance Evaluation",
    subtitle = "Analysis of factors impacting Rad51 yeast mutant survival parameters",
    x = "Telemetry Metrics",
    y = "Node Purity Index (Gini Importance)"
  ) +
  theme_minimal(base_size = 12) +
  theme(
    plot.title = element_text(face = "bold", size = 14),
    panel.grid.minor = element_blank()
  )

# Save visualization report to output directory
ggsave("output/feature_importance_report.png", width = 8, height = 5, dpi = 300)
print("Data visualization report generated and exported to 'output/feature_importance_report.png'")