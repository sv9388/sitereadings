from sqlalchemy import create_engine
from sqlalchemy.schema import MetaData

class SqlEngine():
  def __init__(self, db_uri):
    self.engine = create_engine(db_uri)
    self.conn = self.engine.connect()
    self.meta = MetaData()
    self.meta.reflect(bind=self.engine)

  def get_all_rows(self, tbl_name):
    return [x for x in self.engine.execute(self.meta.tables[tbl_name].select())]

  def add_or_update(self, tbl_name, row_dict):
    print("Old Meta", self.get_all_rows(tbl_name))
    self.conn.execute(self.meta.tables[tbl_name].insert(row_dict))
    print("New Meta", self.get_all_rows(tbl_name))
    return  
