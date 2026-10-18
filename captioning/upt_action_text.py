"""Explicit text for all 116 positive HICO action labels; no new action inference."""

ACTION_TEXT = dict(line.split('=', 1) for line in '''adjust=adjusting
assemble=assembling
block=blocking
blow=blowing toward
board=boarding
break=breaking
brush_with=brushing with
buy=buying
carry=carrying
catch=catching
chase=chasing
check=checking
clean=cleaning
control=controlling
cook=cooking
cut=cutting
cut_with=cutting with
direct=directing
drag=dragging
dribble=dribbling
drink_with=drinking from
drive=driving
dry=drying
eat=eating
eat_at=eating at
exit=exiting
feed=feeding
fill=filling
flip=flipping
flush=flushing
fly=flying
greet=greeting
grind=grinding on
groom=grooming
herd=herding
hit=hitting
hold=holding
hop_on=hopping on
hose=hosing
hug=hugging
hunt=hunting
inspect=inspecting
install=installing
jump=jumping
kick=kicking
kiss=kissing
lasso=lassoing
launch=launching
lick=licking
lie_on=lying on
lift=lifting
light=lighting
load=loading
lose=losing
make=making
milk=milking
move=moving
open=opening
operate=operating
pack=packing
paint=painting
park=parking
pay=paying at
peel=peeling
pet=petting
pick=picking
pick_up=picking up
point=pointing
pour=pouring from
pull=pulling
push=pushing
race=racing
read=reading
release=releasing
repair=repairing
ride=riding
row=rowing
run=running
sail=sailing
scratch=scratching
serve=serving
set=setting
shear=shearing
sign=signing
sip=sipping from
sit_at=sitting at
sit_on=sitting on
slide=sliding on
smell=smelling
spin=spinning
squeeze=squeezing
stab=stabbing
stand_on=standing on
stand_under=standing under
stick=sticking
stir=stirring
stop_at=stopping at
straddle=straddling
swing=swinging
tag=tagging
talk_on=talking on
teach=teaching
text_on=texting on
throw=throwing
tie=tying
toast=toasting with
train=training
turn=turning
type_on=typing on
walk=walking
wash=washing
watch=watching
wave=waving at
wear=wearing
wield=wielding
zip=zipping'''.splitlines())

assert len(ACTION_TEXT) == 116 and 'no_interaction' not in ACTION_TEXT


def action_predicate(verb):
    return {'hold': 'holding', 'ride': 'riding'}.get(verb, verb)


def uniform_policy(threshold=.5):
    import math
    if not math.isfinite(threshold) or not 0 < threshold <= 1:
        raise ValueError('Exploratory threshold must be in (0,1].')
    return {'mode': 'all_actions_exploratory', 'calibrated': False,
            'predicates': {action_predicate(v): {'threshold': threshold} for v in ACTION_TEXT}}
