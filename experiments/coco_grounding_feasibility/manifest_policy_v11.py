"""One documented, stricter correction after the original fixed16 mapping review."""
import re
import runpy
from pathlib import Path
from manifest_policy import POLICY as V1,context_exclusion as old_context

# Explicit singular/plural pairs; runtime matching never infers number from suffix 's'.
PAIR_TEXT='''person|people,persons
man|men
woman|women
boy|boys
girl|girls
child|children
kid|kids
rider|riders
bicycle|bicycles
bike|bikes
car|cars
vehicle|vehicles
automobile|automobiles
motorcycle|motorcycles
motorbike|motorbikes
airplane|airplanes
plane|planes
jet|jets
bus|buses
train|trains
truck|trucks
boat|boats
ship|ships
traffic light|traffic lights
traffic signal|traffic signals
fire hydrant|fire hydrants
stop sign|stop signs
parking meter|parking meters
bench|benches
bird|birds
cat|cats
dog|dogs
horse|horses
sheep|sheeps
cow|cows
elephant|elephants
bear|bears
zebra|zebras
giraffe|giraffes
backpack|backpacks
umbrella|umbrellas
handbag|handbags
purse|purses
tie|ties
suitcase|suitcases
frisbee|frisbees
ski|skis
snowboard|snowboards
sports ball|sports balls
ball|balls
kite|kites
baseball bat|baseball bats
baseball glove|baseball gloves
glove|gloves
skateboard|skateboards
surfboard|surfboards
tennis racket|tennis rackets
racket|rackets
racquet|racquets
bottle|bottles
wine glass|wine glasses
cup|cups
fork|forks
knife|knives,knifes
spoon|spoons
bowl|bowls
banana|bananas
apple|apples
sandwich|sandwiches
orange|oranges
broccoli|broccolis
carrot|carrots
hot dog|hot dogs
pizza|pizzas
donut|donuts
doughnut|doughnuts
cake|cakes
chair|chairs
couch|couches
sofa|sofas
potted plant|potted plants
plant|plants
bed|beds
dining table|dining tables
table|tables
toilet|toilets
tv|tvs
television|televisions
monitor|monitors
laptop|laptops
mouse|mice,mouses
remote|remotes
keyboard|keyboards
cell phone|cell phones
phone|phones
mobile phone|mobile phones
microwave|microwaves
oven|ovens
toaster|toasters
sink|sinks
refrigerator|refrigerators
fridge|fridges
book|books
clock|clocks
vase|vases
scissor|scissors
teddy bear|teddy bears
stuffed bear|stuffed bears
hair drier|hair driers
hair dryer|hair dryers
toothbrush|toothbrushes'''
inventory={' '.join(a) for a,_ in runpy.run_path(str(Path(__file__).resolve().parents[2]/'captioning/object_consistency.py'))['ALIASES']}
pairs={s:[p for p in values.split(',') if p in inventory] for s,values in (line.split('|') for line in PAIR_TEXT.splitlines())}
plural=sorted({p for values in pairs.values() for p in values})
ambiguities=set(V1['ambiguity_exclusions'])
for singular,values in pairs.items():
    if singular in ambiguities or set(values)&ambiguities:
        ambiguities.update(p for p in [singular,*values] if p in inventory)

POLICY=dict(V1,version='coco-noun-box-v1.1',ambiguity_exclusions=sorted(ambiguities),
    manual_selection='Reuse the original16 v1 hash-selected records, including removed cases; report keep/rejection effects without replacement.',
    plural_alias_pairs=pairs,plural_surfaces=plural,paired_surface_exceptions=['skis','scissors'],
    colors='black white red blue green yellow orange purple pink brown gray grey beige tan gold golden silver multicolored multicoloured'.split(),
    color_lookback=3,color_boundaries='and or but in on at by with without of to from for under over above below behind beside between near against'.split(),
    color_punctuation_boundaries='.!?,;:',
    multiple_quantifiers=['many','several','multiple','numerous'],
    multiple_unit_patterns=['pairs of','sets of','groups of','lots of','lot of'],
    numeral_values=dict(two=2,three=3,four=4,five=5,six=6,seven=7,eight=8,nine=9,ten=10),
    correction_scope='One stricter v1.1 pass approved after original fixed16; same inputs/gates/sample, no per-image exclusion or threshold relaxation.')


def context_exclusion(tokens,start,end,label,raw_tokens=None):
    reason=old_context(tokens,start,end,label)
    if reason:return reason
    surface=' '.join(tokens[start:end]);before=tokens[max(0,start-4):start]
    paired=surface in POLICY['paired_surface_exceptions']
    if surface in POLICY['plural_surfaces'] and not paired:return 'plural_surface_single_annotation'
    preceding=before[-1] if before else ''
    quantity=POLICY['numeral_values'].get(preceding,int(preceding) if re.fullmatch('[0-9]+',preceding) else 0)
    limit=2 if surface=='skis' else 1
    if preceding in POLICY['multiple_quantifiers'] or quantity>limit:
        return 'explicit_multiple_units'
    if any(' '.join(before[-len(p.split()):])==p for p in POLICY['multiple_unit_patterns']):
        return 'explicit_multiple_units'
    raw=raw_tokens if raw_tokens is not None else tokens
    for j in range(start-1,max(-1,start-POLICY['color_lookback']-1),-1):
        if any(p in raw[j] for p in POLICY['color_punctuation_boundaries']):break
        if tokens[j] in POLICY['color_boundaries']:break
        if set(re.findall('[a-z]+',raw[j]))&set(POLICY['colors']):return 'local_color_modifier'
    return None
