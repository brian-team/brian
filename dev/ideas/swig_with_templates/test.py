from templatecontainer import TemplateContainerInt, TemplateContainerDouble

t = TemplateContainerInt(5)
print t.x
t.x = 7
print t.x

s = TemplateContainerDouble(3.4)
print s.x

u = TemplateContainerDouble(3)
print u.x
