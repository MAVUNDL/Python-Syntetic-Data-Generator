from pipeline.pipeline import Pipeline

pipeline = Pipeline.from_config("config\\config.yaml")

df = pipeline.run(n_rows=1000, seed=42)

df.to_csv("data\\dataset.csv")