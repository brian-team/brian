import os

os.chdir('../../../docs_sphinx')

import brian.globalprefs as gp

main_preferences_page_text=open('reference-preferences.txt', 'r').read()
main_preferences_page_text=main_preferences_page_text[:main_preferences_page_text.find('.. INSERT_GLOBAL_PREFERENCES_HERE')]
main_preferences_page_text+='.. INSERT_GLOBAL_PREFERENCES_HERE\n\n'
main_preferences_page_text+=gp.__doc__

f=open('reference-preferences.txt', 'w')
f.write(main_preferences_page_text)
f.close()
