class Config(object):
  DEBUG = False
  TESTING = False
  SQLALCHEMY_DATABASE_URI = "postgres://powersines:powersines@powersinesdb.cm6zndedailb.eu-central-1.rds.amazonaws.com:5432/siter"
  SQLALCHEMY_ECHO = True
  SECRET_KEY = '\x13\nf\xa0\xe6\xc0\xd7\xa8\xb8.\xbf\xc8\x99NgfZ\xe5S\x822\x84\x02\x01'

class ProductionConfig(Config):
  pass

class DevelopmentConfig(Config):
  DEBUG = True
  SQLALCHEMY_ECHO = False

class TestingConfig(Config):
  TESTING = True
