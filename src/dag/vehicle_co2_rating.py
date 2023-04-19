from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.bash_operator import BashOperator
from airflow.operators.python_operator import PythonOperator


def set_repo_path(**context):
    context['ti'].xcom_push(key='repo_path', value='/Users/macpro/Documents/GitHub/fuel-electric-hybrid-vehicle-ml')  # Replace with the actual path to your repo


default_args = {
    'owner': 'your_name',
    'depends_on_past': False,
    'start_date': datetime(2022, 1, 1),
    'email': ['your_email@example.com'],
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

dag = DAG(
    'vehiche_co2_rating_pipeline',
    default_args=default_args,
    description='Data pipeline to predict vehicle CO2 rating',
    schedule_interval=timedelta(days=1),
    start_date=datetime(2022, 1, 1),
    catchup=False
)


set_repo_path_task = PythonOperator(
    task_id='set_repo_path',
    python_callable=set_repo_path,
    dag=dag
)

data_extraction_task = BashOperator(
    task_id='data_extraction',
    bash_command='python /Users/macpro/Documents/GitHub/fuel-electric-hybrid-vehicle-ml/src/data/data_extraction.py',
    dag=dag
)

train_model_task = BashOperator(
    task_id='train_model',
    bash_command='python /Users/macpro/Documents/GitHub/fuel-electric-hybrid-vehicle-ml/src/models/train_model.py',
    dag=dag
)

predict_model_task = BashOperator(
    task_id='predict_model',
    bash_command='python /Users/macpro/Documents/GitHub/fuel-electric-hybrid-vehicle-ml/src/models/predict_model.py',
    dag=dag
)


# Set task dependencies
set_repo_path_task >> data_extraction_task >> train_model_task >> predict_model_task
