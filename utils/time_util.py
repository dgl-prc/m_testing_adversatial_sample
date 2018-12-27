import time

def sec2str(timestamp,format):
    '''
    :param timestamp: "%Y-%m-%d %H:%M:%S"
    :param formsat:
    :return:
    '''
    if int(timestamp) < 0:
        return ""
    return str(time.strftime(format,time.localtime(timestamp)))

def current_timestamp():
    timestamp = time.time()
    return sec2str(timestamp,"%Y-%m-%d %H:%M:%S")

if __name__=="__main__":
    a=current_timestamp()
    import os
    print a.replace(" ","_")
    # os.mkdir(a.replace(" ","_"))