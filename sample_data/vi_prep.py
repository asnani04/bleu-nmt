import numpy as np
import re
import string


regex = re.compile('[%s]' % re.escape(string.punctuation))


def remove_puncts(filename, extension):
    f1 = open(filename + "." + extension, "rb")
    f2 = open(filename + "_prep." + extension, "wb")
    
    for line in f1.readlines():
        new_line = regex.sub('', line)
        new_line = re.compile('  ').sub(' ', new_line)
        f2.write(new_line)
        
    f1.close()
    f2.close()

remove_puncts("train", "en")
remove_puncts("tst2012", "en")
remove_puncts("tst2013", "en")
remove_puncts("train", "vi")
remove_puncts("tst2012", "vi")
remove_puncts("tst2013", "vi")
