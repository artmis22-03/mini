x=input("enter the symptoms:")
s=x.split(',')
ns=[]
for l in s:
    if l=='stomach pain':
        l='stomach_pain'
        ns.append(l)
    elif l=='skin rash':
            l='skin_rash'
            ns.append(l)
    elif l=='continuous sneezing':
         l='continuous_sneezing'
         ns.append(l)
    elif l=='watering from eyes':
        l='watering_from_eyes'
        ns.append(l) 
    elif l=='nodal skin eruptions':
         l='nodal_skin_eruptions'
         ns.append(l)
    elif l=='chest pain':
         l='chest_pain'
         ns.append(l)
    elif l=='dischromatic patches':
         l='dischromatic_patches'
         ns.append(l)
    elif l=='ulcers on tongue':
         l='ulcers_on_tongue'
         ns.append(l)
    elif l=='yellowish skin':
         l='yellowish_skin'
         ns.append(l)
    elif l=='loss of appetite':
         l='loss_of_appetite'
         ns.append(l)
    elif l=='abdominal pain':
         l='abdominal_pain'
         ns.append(l)
    elif l=='yellowing of eyes':
         l='yellowing_of_eyes'
         ns.append(l)
    elif l=='spotting urination':
         l='spotting_urination'
         ns.append(l)
    elif l=='interna itching':
         l='internal_itching'
         ns.append(l)
    elif l=='high fever':
         l='high_fever'
         ns.append(l)