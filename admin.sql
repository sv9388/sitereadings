drop database siter;
drop user siteruser;
create database siter;
create user siteruser with password 'siteruser';
REVOKE ALL PRIVILEGES on database siter from postgres;
GRANT ALL PRIVILEGES on database siter to siteruser;
