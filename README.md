Package to clean datasets

# TODO
* implement unit tests
* data dictionary for each measure
* include function for aggregating 
  * output as wide
* option for how to handle missing
* visualization module?
  * missingno
* html report at the end that summarizes the steps & consort, etc
* ~ BIDS format
* option to not have any string values
* automate adding modules to __init__

# NOTE
* since `gettrt` and `getcrt` do not overlap much across integneuro & webneuro, we define `getrt` which uses whatever rt column is available (assuming the rt between the conditions are similar enough)
  * ispot: `gettrt` only
  * rad: `getcrt` only
  * conn: `getcrt` only
  * engage: `gettrt` and `getcrt`
* same with `dgttrt` and `dgtcrt`
  * ispot: `crt` only
  * rad: `trt` only
  * conn: `trt` only
  * engage: `crt` and `trt`
