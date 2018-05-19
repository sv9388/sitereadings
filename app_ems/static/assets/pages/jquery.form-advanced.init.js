jQuery('#datepicker').datepicker();
jQuery('#datepicker-autoclose').datepicker({
  autoclose: true,
  todayHighlight: true
});
jQuery('#datepicker-inline').datepicker();
jQuery('#datepicker-multiple-date').datepicker({
  format: "mm/dd/yyyy",
  clearBtn: true,
  multidate: true,
  multidateSeparator: ","
});
jQuery('#date-range').datepicker({
  toggleActive: true
});
//Date range picker
$('.input-daterange-datepicker').daterangepicker({
  buttonClasses: ['btn', 'btn-sm'],
  applyClass: 'btn-secondary',
  cancelClass: 'btn-primary'
});
$('.input-daterange-timepicker').daterangepicker({
  timePicker: true,
  format: 'MM/DD/YYYY h:mm A',
  timePickerIncrement: 30,
  timePicker12Hour: true,
  timePickerSeconds: false,
  buttonClasses: ['btn', 'btn-sm'],
  applyClass: 'btn-secondary',
  cancelClass: 'btn-primary'
});
$('.input-limit-datepicker').daterangepicker({
  format: 'MM/DD/YYYY',
  minDate: '06/01/2016',
  maxDate: '06/30/2016',
  buttonClasses: ['btn', 'btn-sm'],
  applyClass: 'btn-secondary',
  cancelClass: 'btn-primary',
  dateLimit: {
    days: 6
  }
});

$('#reportrangesingle span').html(moment().subtract(29, 'days').format('YYYY-MM-DD') + ' - ' + moment().format('YYYY-MM-DD'));
$('#reportrangesingle').daterangepicker({
      locale: {
        format: 'YYYY-MM-DD'
      },
      "showDropdowns": true,
      "ranges": {
        "Last 7 Days": [ moment().subtract(7, "days").format("YYYY-MM-DD"),  moment().format("YYYY-MM-DD") ],
        "Last 30 Days": [ moment().subtract(30, "days").format("YYYY-MM-DD"),  moment().format("YYYY-MM-DD") ],
        "Current Month" : [ moment().year(moment().year()).month(moment().month()).date(1).format("YYYY-MM-DD"),  moment().format("YYYY-MM-DD") ],
        "Last 12 Months": [ moment().subtract(12, "months").format("YYYY-MM-DD"), moment().format("YYYY-MM-DD") ],
        "Current Year" : [  moment().year(moment().year()).month(1).date(1).format("YYYY-MM-DD"), moment().format("YYYY-MM-DD") ],
        "All" : [ moment().subtract(5, "years").format("YYYY-MM-DD"), moment().format("YYYY-MM-DD") ]
      },
      "startDate": moment().subtract(7, "days").format("YYYY-MM-DD"),
      "endDate": moment().format("YYYY-MM-DD"),
  opens: 'left',
  drops: 'down',
  buttonClasses: ['btn', 'btn-sm'],
  applyClass: 'btn-success',
  cancelClass: 'btn-secondary'

}, function(start, end, label) {
  console.log(start.toISOString(), end.toISOString(), label);
  $('#reportrangesingle span').html(start.format('YYYY-MM-DD') + ' - ' + end.format('YYYY-MM-DD'));
});

$('#secondreportrange span').html(moment().subtract(29, 'days').format('YYYY-MM-DD') + ' - ' + moment().format('YYYY-MM-DD'));
$('#secondreportrange').daterangepicker({
      locale: {
        format: 'YYYY-MM-DD'
      },
      "showDropdowns": true,
      "ranges": {
        //"None" : ["", ""],
        "Previous 7 days": [ moment().subtract(14, "days").format("YYYY-MM-DD"),  moment().subtract(7, "days").format("YYYY-MM-DD") ],
        "Previous 30 days": [ moment().subtract(60, "days").format("YYYY-MM-DD"),  moment().subtract(30, "days").format("YYYY-MM-DD") ]
      },
      "startDate": "Invalid Date", //moment().subtract(7, "days").format("YYYY-MM-DD"),
      "endDate": "Invalid Date", //moment().format("YYYY-MM-DD"),
  opens: 'left',
  drops: 'down',
  buttonClasses: ['btn', 'btn-sm'],
  applyClass: 'btn-success',
  cancelClass: 'btn-secondary'
}, function(start, end, label) {
  console.log(start.toISOString(), end.toISOString(), label);
  $('#secondreportrange span').html(start.format('YYYY-MM-DD') + ' - ' + end.format('YYYY-MM-DD'));
});

$('#firstreportrange span').html(moment().subtract(29, 'days').format('YYYY-MM-DD') + ' - ' + moment().format('YYYY-MM-DD'));
$('#firstreportrange').daterangepicker({
      locale: {
        format: 'YYYY-MM-DD'
      },
      "showDropdowns": true,
      "ranges": {
        "Last 7 Days": [ moment().subtract(7, "days").format("YYYY-MM-DD"),  moment().format("YYYY-MM-DD") ],
        "Last 30 Days": [ moment().subtract(30, "days").format("YYYY-MM-DD"),  moment().format("YYYY-MM-DD") ],
        "Current Month" : [ moment().year(moment().year()).month(moment().month()).date(1).format("YYYY-MM-DD"),  moment().format("YYYY-MM-DD") ],
        "Last 12 Months": [ moment().subtract(12, "months").format("YYYY-MM-DD"), moment().format("YYYY-MM-DD") ],
        "Current Year" : [  moment().year(moment().year()).month(1).date(1).format("YYYY-MM-DD"), moment().format("YYYY-MM-DD") ],
        "All" : [ moment().subtract(5, "years").format("YYYY-MM-DD"), moment().format("YYYY-MM-DD") ]
      },
      "startDate": moment().subtract(7, "days").format("YYYY-MM-DD"),
      "endDate": moment().format("YYYY-MM-DD"),
  opens: 'left',
  drops: 'down',
  buttonClasses: ['btn', 'btn-sm'],
  applyClass: 'btn-success',
  cancelClass: 'btn-secondary'
}, function(start, end, label) {
  console.log(start.toISOString(), end.toISOString(), label);
  $('#firstreportrange span').html(start.format('YYYY-MM-DD') + ' - ' + end.format('YYYY-MM-DD'));
});
