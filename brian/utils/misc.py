# Could not find a better file name!
import urllib,re

__all__=['brian_downloads']

def brian_downloads():
    """
    Returns the total number of Brian downloads.
    """
    text=urllib.urlopen('http://gforge.inria.fr/top/toplist.php?type=downloads').read()
    return int(re.search(r'Brian.*?(\d+)',text).group(1))

if __name__=='__main__':
    print brian_downloads()