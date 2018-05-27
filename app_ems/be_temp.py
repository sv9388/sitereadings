@app.route("/delltr")
def bulk_edit():
  import pandas as pd
  from sqlalchemy import create_engine
  df = pd.read_csv("./new_data.csv")
  df['is_active'] = df['is_active'] == "Y"
  df.device_id = df.device_id.apply(str)
  engine = create_engine("postgresql+psycopg2://powersines:powersines@powersinesdb.cm6zndedailb.eu-central-1.rds.amazonaws.com:5432/siter")
  df.to_sql('temp_table', engine, if_exists='replace')

  sql = """UPDATE device AS f SET distributer_name = t.distributer_name, project = t.project, system_name = t.system_name, device_unique_name = t.device_unique_name, is_active = t.is_active FROM temp_table AS t WHERE f.device_id = t.device_id"""

  with engine.begin() as conn:     # TRANSACTION
    conn.execute(sql)
  return jsonify(message = "Updated")

