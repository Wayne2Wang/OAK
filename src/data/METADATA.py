METADATA = {

    'stanford_alm': {
        'all_taxonomies': {
            'action' :  ['applauding', 'blowing_bubbles', 'brushing_teeth', 'cleaning_the_floor', 'climbing', 'cooking', 'cutting_trees', 'cutting_vegetables', 'drinking', 'feeding_a_horse', 'fishing', 'fixing_a_bike', 'fixing_a_car', 'gardening', 'holding_an_umbrella', 'jumping', 'looking_through_a_microscope', 'looking_through_a_telescope', 'phoning', 'playing_guitar', 'playing_violin', 'pouring_liquid', 'pushing_a_cart', 'reading', 'riding_a_bike', 'riding_a_horse', 'rowing_a_boat', 'running', 'shooting_an_arrow', 'smoking', 'taking_photos', 'texting_message', 'throwing_frisby', 'using_a_computer', 'walking_the_dog', 'washing_dishes', 'watching_TV', 'waving_hands', 'writing_on_a_board', 'writing_on_a_book'],
            'location' : ['educational_institution', 'natural_environment', 'office_or_workplace', 'public_event_or_gathering', 'residential_area', 'restaurant_or_dining_area', 'sports_facility', 'store_or_market', 'transportation_hub', 'urban_area_or_city_street'],
            'mood' : ['adventurous', 'focused', 'joyful', 'relaxed'],
            'location_v2' : ['educational_institution', 'natural_environment', 'office_or_workplace', 'public_event_or_gathering', 'residential_area', 'restaurant_or_dining_area', 'sports_facility', 'store_or_market', 'transportation_hub', 'urban_area_or_city_street'],
            'mood_v2' : ['adventurous', 'caring', 'creative', 'focused', 'playful', 'social', 'productive', 'reflective', 'relaxed']
        },
        'train_classes': {
            'action' : list(range(0, 40, 2)),
            'location' : list(range(5)),
            'mood' : list(range(0, 4, 2)),
            'location_v2' : list(range(5)),
            'mood_v2' : list(range(5)),
        }
    },

    'clevr4': {
        'all_taxonomies': {
            'color': ["gray", "red", "blue", "green", "brown", "purple", "cyan", "yellow", "pink", "orange"],
            'texture': ["rubber", "metal", "checkered", "emojis", "wave", "brick", "star", "circles", "zigzag", "chessboard"],
            'count': ["1", "2", "3", "4", "5", "6", "7", "8", "9", "10"],
            'shape': ["cube", "sphere", "monkey", "cone", "torus", "star", "teapot", "diamond", "gear", "cylinder"]
        },
        'train_classes': {
            'color' : [0, 1, 2, 3, 4],
            'texture' : [0, 1, 2, 3, 4],
            'count' : [6, 9, 0, 2, 4],
            'shape' : [0, 1, 2, 3, 4]
        }
    }

}