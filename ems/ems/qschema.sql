\c siter;
drop table reading;
drop table device;
create table device (
  id serial PRIMARY KEY,
  device_id varchar(10)  NOT NULL
);

create table reading (
  id serial PRIMARY KEY,
  rdate timestamp unique not null,
  total_power_kw float not null,
  total_kwh float not null,
  device_id integer references device(id)
);
