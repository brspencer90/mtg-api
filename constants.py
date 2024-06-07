import pandas as pd

class Constants():
    col = ['Name','Mana Cost','CMC','Power','Toughness','Type','Creature','Non-Creature','Land','Text','Colours','Colour Identity','Keywords','Rarity','Collector No.','Price','Price STD','Price Foil','Price Etched']

    list_kw = ['Enchant', 'Collect evidence', 'Flying', 'Double strike', 'Haste', 'Mill', 'Hexproof from', 'Trample', 'Vigilance', 'Defender', 'Landcycling',
                'Disguise', 'Basic landcycling', 'Ward', 'Investigate', 'Suspect', 'Cloak', 'Deathtouch', 'Indestructible', 'Cycling', 'Menace', 'Prowess',
                'Hexproof', 'Surveil', 'Reach', 'Crew', 'Equip', 'Typecycling', 'Lifelink', 'Flash']

    list_crea_type = ['Devil', 'Artificer', 'God', 'Dog', 'Wizard', 'Hellion', 'Horror', 'Gargoyle', 'Legendary', 'Spirit', 'Octopus', 'Troll', 'Beast', 'Goblin',
                    'Ogre', 'Viashino', 'Insect', 'Goat', 'Faerie', 'Crocodile', 'Advisor', 'Wall', 'Skeleton', 'Artifact', 'Fish', 'Phoenix', 'Wurm',
                        'Giant', 'Zombie', 'Elephant', 'Weird', 'Plant', 'Clue', 'Demon', 'Centaur', 'Cyclops', 'Gorgon', 'Bird',  'Human', 'Druid', 'Shaman',
                        'Soldier', 'Golem', 'Leech', 'Detective', 'Citizen', 'Avatar', 'Assassin', 'Cat', 'Drake', 'Merfolk', 'Shapeshifter', 'Construct',
                        'Warrior', 'Thopter', 'Ooze', 'Spider', 'Elemental', 'Vampire', 'Rogue', 'Lammasu', 'Dragon', 'Angel', 'Bard', 'Archon', 'Turtle',
                        'Homunculus', 'Scout', 'Cleric', 'Thrull', 'Unicorn', 'Ape', 'Sphinx', 'Elf', 'Elk', 'Mole', 'Archer', 'Vedalken', 'Dryad']

    dict_colour = {'B':'Black','W':'White','R':'Red','G':'Green','U':'Blue','N':'Uncoloured'}
    dict_colour_map = {'Black':'slategray','White':'gainsboro','Red':'crimson','Blue':'royalblue','Green':'seagreen','Uncoloured':'wheat'}
    list_colour_abbr = ['B','W','R','G','U','N']
    list_colour = ['Black','White','Red','Green','Blue','Uncoloured']

    list_rarity = ['uncommon', 'common', 'rare', 'mythic']

    list_card_type = ['Creature','Non-Creature','Land']

    col_gf = ['Card','Set ID','Set Name','Quantity','Foil','Variation']

    dict_scores = {
        'A+':4.3,
        'A':4.0,
        'A-':3.7,
        'B+':3.3,
        'B':3.0,
        'B-':2.7,
        'C+':2.3,
        'C':2.0,
        'C-':1.7,
        'D+':1.3,
        'D':1.0,
        'D-':0.7,
        'F':0.0    
        }

    on_crime_regex_list = [
                    r'(\+\d\/\+\d counter)[\s\S]*(target)',
                    r'(target)[\s\S]*(your graveyard)', 
                    r'(target)[\s\S]*(gains)'
                ]

    expansion_list = list(pd.read_csv('card_data.csv')['expansion'].unique())