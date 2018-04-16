from wtforms.fields import SelectField, SelectMultipleField
from wtforms.validators import ValidationError
from wtforms.widgets import HTMLString, html_params
from cgi import escape
from wtforms.widgets import Select

# very loosely based on https://gist.github.com/playpauseandstop/1590178

__all__ = ('OptgroupSelectField', 'OptgroupSelectWidget')


class OptgroupSelectWidget(Select):
  """
  Add support of choices with ``optgroup`` to the ``Select`` widget.
  """
  def __call__(self, field, **kwargs):
    kwargs.setdefault('id', field.id)
    if self.multiple:
      kwargs['multiple'] = True
    html = ['<select %s>' % html_params(name=field.name, **kwargs)]
    print(field.choices)
    for item1, item2 in field.choices:
      if isinstance(item2, (list,tuple)):
        group_label = item1
        group_items = item2
        html.append('<optgroup %s>' % html_params(label=group_label))
        for inner_val, inner_label in group_items:
          html.append(self.render_option(inner_val, inner_label, inner_val == field.data))
        html.append('</optgroup>')
      else:
        val = item1
        label = item2
        html.append(self.render_option(val, label, val == field.data))
    html.append('</select>')
    return HTMLString(''.join(html))


class OptgroupSelectField(SelectField):
  widget = OptgroupSelectWidget()

  def pre_validate(self, form):
    for item1,item2 in self.choices:
      if isinstance(item2, (list, tuple)):
        group_label = item1
        group_items = item2
        for val,label in group_items:
          if val in self.data:
            return
      else:
        val = item1
        label = item2
        if val == self.data:
          return
    raise ValueError(self.gettext('Not a valid choice!'))

class OptgroupSelectMultipleField(SelectMultipleField):
  widget = OptgroupSelectWidget(multiple = True)
  def pre_validate(self, form):
    for item1,item2 in self.choices:
      if isinstance(item2, (list, tuple)):
        group_label = item1
        group_items = item2
        for val,label in group_items:
          if val in self.data:
            return
      else:
        val = item1
        label = item2
        if val == self.data:
          return
    raise ValueError(self.gettext('Not a valid choice!'))


