
# Web Crawler project
This script crawls foundation data ,**time** and **accumulated net value**, from [天天基金](http://fund.eastmoney.com/), and gives you a prediction of how accumulated value will be the next day by the end of the plot.

The data is from a period between the time you run the script and 200 days ago.

Have a look at the help documention of this script parameters:

> \>python predict.py --help
>       usage: predict.py [-h] fcode

> Predict foundation accumulate net value

>positional arguments:
	    fcode       foundation id to predict

> optional arguments:
  -h, --help  show this help message and exit


  Concretly,run this script like this :

>  python predict.py 001371

  to predict a foundation of which id is  **001371**.

## Requirement for this script
- Keras
- Pandas

