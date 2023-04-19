To restart all Airflow processes, you can follow these steps:

Stop all Airflow processes:

```
airflow webserver -p 8080 --stop
airflow scheduler --stop
pkill -f "airflow worker"
pkill -f "airflow triggerer"
```

Wait a few seconds to make sure all processes have stopped.

Start the Airflow processes again:

```
airflow db init
airflow webserver -p 8080
airflow scheduler
airflow worker
airflow triggerer
```