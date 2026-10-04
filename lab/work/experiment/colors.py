import re
# Derived from each annotation's own clothing description; reviewers sometimes swap generic IDs.
def clothing_color(desc):
 text=desc.lower().split(';')[0]
 match=re.search(r'(green|yellow|red(?:-and-black)?|black|gray|grey|dark(?: gray/black| gray)?)[\w /-]*shorts',text)
 if not match:raise ValueError('Cannot identify shorts color: '+desc)
 head=match.group(1)
 return 'Green' if head=='green' else 'Yellow' if head=='yellow' else 'Red' if head.startswith('red') else 'Black/gray'
def colorize(annotation):
 import json
 x=json.loads(json.dumps(annotation));fighters=x.get('fighters',{});mapping={k:clothing_color(v) for k,v in fighters.items()}
 def rewrite(z):
  if isinstance(z,str):
   for old,new in mapping.items():z=re.sub(r'\b'+re.escape(old)+r'\b',new,z)
   return z
  if isinstance(z,list):return [rewrite(v) for v in z]
  if isinstance(z,dict):return {mapping.get(k,k):rewrite(v) for k,v in z.items()}
  return z
 return rewrite(x)
