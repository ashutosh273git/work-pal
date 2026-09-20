from src.evaluation import run_evaluation


def main():
    print("Running HR policy assistant evaluation...")
    results = run_evaluation()
    print("Done. Open your LangSmith project to see the experiment.")
    print(results)


if __name__ == "__main__":
    main()
    
# future version i have to seperate judge and mainn llm