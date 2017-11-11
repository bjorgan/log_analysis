# log_analysis
Scripts for analysis/plots of ham radio logs, closely related to a series of blog posts at la1k.no. Currently made for analysis of the logs made during LA1K's use of the callsign LM100UKA during the UKA festival (http://uka.no). 

The plots that are (currently) generated can be viewed at https://www.la1k.no/?p=1578. Most are general, but one is also very specific for this post (`qso_frequency_per_day(...)` at https://github.com/bjorgan/log_analysis/blob/master/analyze_logs.py#L103). This script should as of now be considered to be an example that can be adapted for other applications.

Synopsis:

`python3 analyze_logs.py [path_to_containing_folder] [station callsign (optional)]`

Will walk through all subfolder, subsubfolders, ..., and extract contacts from the N1MM .s3db log files, and generate .png-plots in the working directory. If the station callsign is specified, it will only consider contacts made using this callsign. 

Example: For the plots in the post linked above, this was run using

`python3 analyze_logs.py ~/lm100uka_logs/ LM100UKA`

, with the logs contained in `lm100uka_logs` in the home folder on my personal computer. 

The functions can also be imported to another Python script using

```
import analyze_logs

qso_data = get_n1mm_logs_in_path('/path/to/log/folder/')
print qso_data.Operator.unique()
...
```

Might become more of a module that could be used in a production situation in the future, and will then be moved to github.com/la1k/. 
