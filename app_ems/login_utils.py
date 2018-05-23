import datetime
from flask_login import LoginManager, login_user, login_required, logout_user, current_user
from forms import *
from app import app, db, request, render_template

login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'index'
from models import *


@app.route("/logout")
def logout():
  logout_user()
  return render_template('login.html', form = LoginForm(), notif = ['info', "Logged out!"])

@app.route('/', methods=['GET','POST'])
def index():
  cu = current_user.get_id()
  print(cu, type(cu))
  if cu:
    return render_template('profile.html', user = load_user(cu), form = EditProfileForm())
  form = LoginForm()
  if request.method == 'GET':
    return render_template('login.html', form=form)
  if request.method == 'POST':
    print(form.validate_on_submit())
    if form.validate_on_submit():
      email = form.email.data
      print(type(email))
      user = Emsuser.query.filter(Emsuser.email == email)
      print(email, user)
      user = user.first()
      print(user, form.errors)
      if user:
        if user.verify_password(form.password.data): 
          login_user(user)
          return render_template('profile.html', form = EditProfileForm(), user = user, notif = ['info', "Logged in!"])
        return render_template('login.html', form=form, notif = ['error', "Wrong password"] )
      return render_template('login.html', form=form, notif = ['error', "user doesn't exist"] )
    return render_template('login.html', form=form, notif = ['error', form.errors])

@login_manager.user_loader
def load_user(email):
    return Emsuser.query.filter_by(email = email).first()
