import datetime
from flask_login import LoginManager, login_user, login_required, logout_user, current_user
from forms import *
from app import app, db, request, render_template

login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'index'
from models import *

@app.route('/', methods=['GET','POST'])
def index():
  cu = current_user.get_id()
  if cu:
    user = Emsuser.query.filter_by(email = cu).first()
    return render_template('profile.html', user = user)
  form = LoginForm()
  if request.method == 'GET':
    return render_template('login.html', form=form)
  if request.method == 'POST':
    print(form.validate_on_submit())
    if form.validate_on_submit():
      user=Emsuser.query.filter_by(email=form.email.data).first()
      print(user)
      if user:
        if user.verify_password(form.password.data): 
          login_user(user)
          return render_template('profile.html', user = user, notif = ['info', "Logged in!"])
        return render_template('login.html', form=form, notif = ['error', "Wrong password"] )
      return render_template('login.html', form=form, notif = ['error', "user doesn't exist"] )
    return render_template('login.html', form=form, notif = ['error', form.errors])

