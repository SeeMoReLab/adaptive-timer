## Learning Agent
To run the program with example scheduled results: `python learning_agent/main.py --schedule-file learning_agent/examples/timeout_schedule.csv`.

## Data Exchange
To run the example experiment aggregation:

`python data_exchange/learning_agent.py --unit 0 --port 50000 --episodes 10`

`python data_exchange/client.py --unit 0 --port 50000 --episodes 10`