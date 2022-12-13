import gym
import numpy as np
from gym import utils
from gym_mujoco.envs.mujoco_env import MujocoEnv
# from stable_baselines3.common.utils import load_pca_transformation_numpy


class ReacherTrackerEnv(MujocoEnv, utils.EzPickle):
    def __init__(self, num_links=10, goal=None,
                slope=None,
                gravity=None,
                mass=None,
                friction=None,
                density=None,
                damping=None,
                native_dim=-1,
                encoder_model=None,
                latent_dim=-1,
                hidden_layer=-1,
                model_path=None,
                ):
        self.encoder_model = encoder_model
        self.native_dim = native_dim
        self.latent_dim = latent_dim
        self.ae_model = None

        if num_links == 10:
            ae_lower_bounds = {
                1: [-0.37403545478000866], 
                2: [0.1868643807721963, -0.5378054873152545], 
                3: [-0.2518686222763531, -0.07595001279042032, -0.22384771034961143], 
                4: [-0.30042004761670493, -0.37834610013689957, -0.11838192815719956, -0.23000780688578926], 
                5: [-0.017678681216734493, 0.011767234829020368, -0.21588145276541634, -0.11207763659575856, -0.5363968156967158], 
                6: [-0.10491040726817014, -0.0776265884492867, -0.1282295598260047, -0.25034533002755677, -0.3554044681532198, -0.2044288686525482], 
                7: [-0.2419193612336054, -0.21869473724928662, -0.1337115595000339, -0.34266625646299304, -0.08242557980022708, -0.26687824266631804, -0.32506918756766645], 
                8: [-0.155698139084214, -0.07627795581058155, -0.0667413965134709, 0.10036355403893141, -0.2861235670412655, -0.21979053850810526, -0.16222003616556085, 0.05308256100778161], 
                9: [-0.07935589193384557, -0.033492120336104894, -0.18096453001910479, -0.03818882013248985, -0.18511889968824297, -0.12411506256711277, -0.019296615766914174, -0.1581254467159972, 0.058797351822126254], 
                10: [-0.22629026758447945, -0.06060808499962099, -0.13667470949204716, -0.18916094950587148, -0.21788388616434562, -0.12772743268711242, -0.0702320713558055, -0.13565533539045502, -0.1249766542032533, -0.20951116893946398]
            }

            ae_upper_bounds = {
                1: [0.07883650428799036], 
                2: [0.884681586970715, 0.07955223663116526], 
                3: [0.2801154758828465, 0.459659049333057, 0.2892969135203809], 
                4: [0.16170859339433014, 0.15068512583056273, 0.3936016260586632, 0.22129729528068137], 
                5: [0.44860792625524676, 0.4387325544901425, 0.15735958944329284, 0.37822114098078774, -0.11622323287553801], 
                6: [0.24280663854823448, 0.24510550677114568, 0.19187164896966952, 0.15317741636372198, 0.10060811244874673, 0.19308304767859738], 
                7: [0.053399533833384985, 0.09574160063359477, 0.16262530976311074, -0.01165710830450084, 0.39136879587304285, 0.054390157620893845, -0.051103210905014296], 
                8: [0.13956988490477465, 0.19424325235101372, 0.2798198855317209, 0.3973514750266396, 0.03012864311008412, 0.11783463644159761, 0.06594162103807195, 0.3106888177548932], 
                9: [0.1193069203356755, 0.22454476394198858, 0.07059156288660148, 0.2005597837870289, 0.1442080598073565, 0.062318526146285434, 0.21541222916660552, 0.1890185827730612, 0.32682717932477323], 
                10: [0.014777233707112161, 0.15729536294621033, 0.11394718389936956, 0.07045030216435175, 0.003918752460835423, 0.09429199760458849, 0.076298024544187, 0.01024394050285505, 0.09059347075911073, 0.04348242173670873], 
            }

            vae_lower_bounds = {
                1: [0.4184459566692339], 
                2: [0.3741033601396535, 0.17708496747270797], 
                3: [-0.0791044408329548, 0.4058234025865595, 0.4586450961365076], 
                4: [0.04899322049595685, 0.17025370613738008, -0.11334384438984987, 0.4703342413552847], 
                5: [0.40469888739066134, 0.4538421841719058, 0.19093118623928435, 0.4312821457307828, 0.16645757629070307], 
                6: [-0.09683512320463292, -0.12433276717035957, 0.23225683593920898, -0.1360330468214731, -0.09587404745818068, -0.1728065938278152], 
                7: [-0.07152246827515281, 0.12239257495876012, -0.1773031582287808, 0.2422829053410071, -0.1060084454778732, -0.1596584503756487, 0.21657360132939688], 
                8: [0.1519230544733064, 0.03922735746454442, 0.4285898439394283, 0.22921841856557862, -0.1321197610529783, -0.09519755760650803, 0.06615666149723506, 0.41439811424274264], 
                9: [0.4854292037220357, 0.3894507262007426, -0.15386230890586255, 0.45280432952424743, 0.3412038241689125, 0.12707643991557105, -0.11686324755685644, -0.15902506109498962, 0.2787497543967926], 
                10: [-0.0985472997137263, 0.08580944500759363, 0.07909045695036337, 0.09956859292994094, -0.15165940334923092, -0.1100023370449566, 0.14071594154799288, -0.12212440438548812, 0.22728777462158453, -0.0978651540985377]
            }

            vae_upper_bounds = {
                1: [0.5600736599410363], 
                2: [0.6882009321038193, 0.42668276271966693], 
                3: [-0.0452039610326629, 0.5260580862029877, 0.7821677838446451], 
                4: [0.2338260467627072, 0.4566617527432951, -0.09360747481494057, 0.6777462277543714], 
                5: [0.5647205511996751, 0.594480618423241, 0.3981806985078907, 0.5381650326277618, 0.45189156414189713], 
                6: [-0.014857421111720823, 0.03952919511853057, 0.4264982795492844, -0.08038211318151821, 0.02281411939853526, -0.04316063764179154], 
                7: [0.02579302910042009, 0.309143722499461, 0.04127097962224595, 0.4321665532907822, -0.0012826906582388345, -0.028286252172564683, 0.380482062951423], 
                8: [0.4689005326205145, 0.21006347598861516, 0.6877285064133177, 0.4573557377562059, 0.0030964516554144172, -0.03928215835186301, 0.2658686822032659, 0.6845399049129629], 
                9: [0.6300867422916477, 0.5903767670844287, 0.07854962308885592, 0.6962263085326263, 0.5208106825582539, 0.3192737254827375, 0.09197410143017618, -0.12724615485553378, 0.4934230234178667], 
                10: [-0.001361215089429102, 0.3018065615485798, 0.2813454384418417, 0.35343576802977317, -0.11812659312114668, -0.040249155154941166, 0.5483563377700422, 0.08613693554541277, 0.47017233975639927, 0.0015590722231715454]
            }
            
            if encoder_model == "OTNAE":
                lower_bounds = [-0.017683667058753072, -0.21571790818650127, -0.1403526412906495, -0.09719098338397907, 0.05449364581454747, -0.21680754149241144, -0.0442904256828364, -0.18480360666719176, -0.15268896707606783, 0.047468896841804414]
                upper_bounds = [0.5080702049547984, 0.10338937844938742, 0.1120389124277937, 0.16888925347173112, 0.1741982586660305, -0.14112092938566323, -0.01098748682054448, -0.13757020783383067, -0.10761855580904756, 0.07812482269954728]
            elif encoder_model == "AE":
                lower_bounds = ae_lower_bounds[latent_dim]
                upper_bounds = ae_upper_bounds[latent_dim]
            elif encoder_model == "VAE":
                lower_bounds = vae_lower_bounds[latent_dim]
                upper_bounds = vae_upper_bounds[latent_dim]
            elif encoder_model == "OTNVAE":
                lower_bounds = [-0.12265798890567356, -0.040877085928190476, 0.2102157158388234, 0.20800077525821384, 0.17149209287612885, -0.3826734789102839, 0.6660696790403893, 0.14259020842414707, -0.16662076273671345, -0.23412331728238453]
                upper_bounds = [0.28489817337199796, 0.4392874065513987, 0.6490014902880086, 0.5930195533217191, 0.4875931225213742, 0.1520273060752064, 0.7784115563566043, 0.46111288563902797, 0.1370310769928115, 0.5877305855544163]
        elif num_links == 20:
            ae_lower_bounds = {
                1: [0.11654356839776847], 
                2: [-0.71457230543081, -0.5190248841018404], 
                3: [-0.2686545100418382, -0.4957260571951172, -0.3493624489798335], 
                4: [-0.1894229624367326, -0.3646895567549489, -0.2730213625293881, -0.46471884481088527], 
                5: [-0.13769191792585453, -0.19384671149771976, -0.00998948794685034, -0.2634629423484426, -0.3035601393989214], 
                6: [-0.24654313159066418, -0.321515263267238, -0.3290438500537451, -0.333872324508557, -0.27403647811181797, -0.08269771069820413], 
                7: [-0.19107386549216468, -0.13128052468168874, -0.37118915069827707, -0.07243730812842972, -0.1811758789112665, -0.34277183228615044, -0.30793952893474125], 
                8: [-0.3581253172744663, -0.09037713245843752, -0.2819618444702425, -0.3054707224748926, -0.17666818082736827, -0.42078964355402626, -0.0899016047154964, -0.14990114426415954], 
                9: [-0.4039592393196594, -0.28583694662350234, -0.17406326310009737, -0.2098445545893487, 0.0005507277955715817, -0.20968097008294684, -0.18783437259603175, -0.2996671669446801, -0.19022781392720303], 
                10: [-0.11539466580149776, -0.048711778194074476, -0.349875647360151, -0.2203751495570059, -0.3233514887991632, -0.25273707190767813, -0.04662535558273033, -0.35871291365743385, -0.1955080652012821, -0.08797569690657489], 
                11: [-0.13112168696817222, 0.03103602597602015, -0.010146024747140298, -0.1587447056537868, -0.16801467508879278, -0.18191029341180273, -0.03374766089342002, 0.07095260933595124, -0.42240443677589923, -0.20656432206393252, -0.43578197071431735], 
                12: [-0.43985686286559883, -0.23523074772030245, -0.120815462919711, -0.24421832401328294, -0.15765477515348822, -0.2861471657764541, -0.09272566432636256, -0.0394962867365988, 0.06689597659121624, -0.09757153699616247, -0.24501927465486747, -0.34624772910330076], 
                13: [-0.11076014381743779, -0.10782985520526553, -0.09227846283651227, -0.29813571190038457, -0.2696534855637237, -0.2061238662172871, -0.18861462565530923, -0.09557522042488395, -0.004847847515829423, -0.04579402857041924, -0.11871929874241374, -0.2586851038951266, -0.2092953869732726], 
                14: [-0.07255466998907331, -0.3460102066039526, 0.0010918463058930128, -0.13103345398359967, -0.10918608312321165, -0.10348118205728363, 0.07344944975410425, -0.15113058790725023, -0.15072375155489146, -0.10255304520242854, -0.1357900669311198, -0.03872661596225191, -0.18711409015835945, -0.13321944579392853], 
                15: [-0.14817796709593958, -0.13430156428524267, -0.06513006618642983, 0.012026225310567545, -0.0308797366913329, -0.11567931434576148, -0.15710160603044504, -0.14318583501994087, -0.22251511455619477, -0.2006963516268934, 0.06214553884171081, -0.13004393005672502, -0.15446272026839022, -0.05679247457666889, -0.24872096899922735], 
                16: [-0.34360873661217356, -0.19704323243721836, -0.2437001590192687, -0.11651834545376942, -0.07010504120919667, -0.32630077776447697, -0.2407590032308136, -0.07839807846680699, -0.30964788995477316, -0.2301113760650409, 0.05930794040901799, -0.22917636531963798, -0.09730817612870143, -0.28299075235053955, -0.3396250194968553, -0.1449054201729022], 
                17: [-0.2617535389975402, -0.22053855331468847, -0.30898075682423853, -0.22760448550416612, -0.13209661780314946, -0.011770599249925268, -0.04845431156529903, -0.20545795138940118, 0.03214526076580124, -0.1656039901998148, -0.058655730611700374, -0.1418366725805144, -0.18141375146357103, 0.07679484964447632, -0.16144853896823635, -0.2390681975294207, -0.251584150629274], 
                18: [-0.13067833533471437, -0.18549925478084062, -0.15901924655006228, -0.06353866295243928, -0.057354682449125255, 0.06570683285388698, -0.06880728377034159, -0.023169083368974858, -0.1569285553620408, -0.29398966361039713, -0.166874490109597, -0.20651512066777483, -0.16733335027170046, -0.1214311866696582, -0.047196589761857574, -0.3218808861783138, -0.12651568035233832, -0.21641282655708058], 
                19: [-0.3171901067466624, -0.11044801794057096, -0.2766407123923108, -0.03734157751678044, -0.35193968684267585, -0.13969197937511363, -0.2769684903529246, -0.1626455647829913, -0.2710316474490119, 0.11114134391240776, -0.139608010685633, -0.1514996826839077, -0.11905179239262174, -0.1644408169485248, 0.060048130402670694, -0.039489126720986034, -0.24588457337161185, -0.2894949112184175, -0.13010232728437832], 
                20: [-0.0026667413850506583, -0.09294085234958581, -0.32195273974929484, -0.1691414797483092, -0.16464679033377436, -0.3639928171557215, 0.09346940275873268, -0.15714923870486863, -0.12612648710979824, -0.11981382413544242, 0.2222137905943617, -0.3400323209957308, -0.13953463256263574, 0.14066117866372638, -0.2655730906716415, -0.26032350368638163, -0.01715351317929792, -0.1431049544965931, -0.21980776866719248, -0.3227584975056795]
            }

            ae_upper_bounds = {
                1: [0.7505671224505296], 
                2: [-0.1347540840861498, 0.40263992456174685], 
                3: [0.2324376992781924, -0.011489053262945054, 0.21276188151606218], 
                4: [0.4826914496846888, 0.17727466410539375, 0.21628522312286014, 0.06068500974101723], 
                5: [0.2792554523152609, 0.3177631461325497, 0.5452098758637542, 0.2546328290964072, 0.23241171994520374], 
                6: [0.3031738842648097, 0.17649914555594343, 0.17128538417600614, 0.14242927204182365, 0.14673352169891726, 0.2872672732704771], 
                7: [0.17773128279870531, 0.215724803709538, 0.1432183992480242, 0.3747363108678318, 0.3200676935218812, 0.23185989305205615, 0.3080019508456538], 
                8: [0.09953772333246311, 0.2695941788565381, 0.0971190585069339, 0.1706591499172784, 0.19453126027590611, 0.13441913573165556, 0.3038959115994885, 0.33189345419624317], 
                9: [0.04004250757773434, 0.16361062384098646, 0.23638186094093705, 0.15279929599100006, 0.40274143269559537, 0.09837832920293371, 0.1505699043350717, 0.0806921161768889, 0.25025798286948686], 
                10: [0.1453743789362842, 0.28154692125841907, -0.007821883091572501, 0.14494071133079423, 0.17937374648245757, 0.2202884830782745, 0.2527908474068197, 0.12523368968782267, 0.23801418442805466, 0.3425961606197051], 
                11: [0.2864180222038192, 0.384437202073229, 0.3065537877315675, 0.17504419520230988, 0.2036528394233502, 0.24153729248991546, 0.4033284953220311, 0.35118371367488044, -0.06737312688587146, 0.21333637992703117, -0.023574688592862714], 
                12: [-0.08914128846222924, 0.1298617366490284, 0.2741985850356754, 0.1823498021326413, 0.20157081425242618, 0.13959969777581793, 0.21067484187701174, 0.29743801951023596, 0.4192091123142855, 0.2569089116183562, 0.06380409003934957, 0.14763834711676058], 
                13: [0.18181166683565259, 0.2107762948121291, 0.303042999533734, 0.05102883892610709, 0.11251559575834791, 0.10524111456306727, 0.2591098504278187, 0.3061931779080786, 0.30589181552990086, 0.2692509250915482, 0.17979342829296246, 0.07851480761670009, 0.08468241359548753], 
                14: [0.28576783350101154, -0.088510135986277, 0.3297445990101311, 0.23610726278033364, 0.15436589686861488, 0.22331331087521866, 0.37132890754854225, 0.13334745460517008, 0.15504863259611928, 0.27598352654216, 0.178713695346777, 0.30910575383637834, 0.22287909704299827, 0.12620872468772137], 
                15: [0.18135841892448223, 0.13524075262139565, 0.22402203710764326, 0.33611003530071737, 0.28494093555185107, 0.1972776264904585, 0.092160814545045, 0.17508880149211353, 0.08641650924707915, 0.18747984120145514, 0.33607008547989203, 0.2540379976008378, 0.17821799190164567, 0.31883985867310416, 0.07705674804325828], 
                16: [-0.13381675083922828, 0.2547961545709846, 0.12123947223524484, 0.26153032515841335, 0.1720649975914827, 0.06240275182216801, 0.043648877582726245, 0.18371780758516476, 0.11539622158537087, 0.01708209594098252, 0.3398875441461239, 0.16944666624506965, 0.25985135527465747, -0.04051018295683738, 0.06716791389087695, 0.1774879815740402], 
                17: [-5.329223741584732e-05, 0.022418585186554235, 0.021780998550964492, 0.22392435135232555, 0.12381527843546403, 0.23890577723409437, 0.33997579605457307, 0.05503445158798129, 0.33383824599641604, 0.06688531616988107, 0.19432124244038462, 0.14654018675270242, 0.12845678401383714, 0.3038480645345279, 0.1815581428889932, 0.11651807081457452, 0.06611040789495552], 
                18: [0.23298351233391532, 0.11184292346939886, 0.13125527083912647, 0.19294356815284622, 0.22427485216581367, 0.3668171453612036, 0.24862220894283701, 0.32684444792804507, 0.1690701400577581, 0.03293927588367168, 0.16327733172101908, 0.10772468859725459, 0.15921288791726915, 0.14209063251479673, 0.21229295421252792, -0.0924664691014134, 0.11693322398053276, 0.15227264135462812], 
                19: [0.1156204728586331, 0.13657180872619407, -0.013171330229031913, 0.2250909266821488, -0.03942880672945287, 0.16157475516572542, 0.02582039376050993, 0.13846649542744013, 0.012462400767380755, 0.34458822086195967, 0.10304752046136886, 0.10420620845213087, 0.1033562768277412, 0.11040219162120812, 0.28175015559934014, 0.1929423017560143, 0.03762666529097386, 0.12019714504134335, 0.22313666594275816], 
                20: [0.2733325546076446, 0.15242916244399893, -0.04249756867550991, 0.09441219196115247, 0.181012531726163, -0.17478230400094452, 0.33366313493282224, 0.11390102128051849, 0.17514770899557658, 0.24083613453032615, 0.4170259708531533, -0.04884437291307919, 0.0963927635293736, 0.34820841594881924, -0.0001158746239085795, -0.03447258490645208, 0.19085076021047218, 0.0812847471143135, 0.09885932418344817, -0.07519430148304374]
            }

            vae_lower_bounds = {
                1: [0.25767739794553035], 
                2: [-0.07550790528613094, 0.27839061927044706], 
                3: [-0.1478769349817246, 0.37917015877843624, -0.17791836474287315], 
                4: [-0.14853035642182752, -0.15125658377334, 0.22687834429586598, -0.06308063459680052], 
                5: [0.4010631200174547, 0.2781040868014093, -0.21497081415808025, 0.3473531139588575, 0.2557176210635177], 
                6: [-0.21599519224411898, -0.1874026392766238, -0.17936988267897577, -0.2215610890166033, 0.33973002414244236, 0.12524351687668228], 
                7: [-0.14192051385178073, -0.20252748745857851, 0.3055576358046327, 0.11857679181067327, 0.2700409866607534, -0.18183702038073332, 0.32850533412946], 
                8: [0.24215801032580672, -0.18678324348878012, 0.2932067110144574, -0.21752722094971175, 0.34270157833379217, -0.10168750635541265, 0.2975748700882659, 0.04120576192842346], 
                9: [0.4120403227555193, -0.18580403643322357, 0.10952122177735213, -0.1390623341675179, -0.1696523346604465, 0.24357672618715523, 0.2999447381034934, 0.03579286579770721, 0.25857322869722765], 
                10: [-0.20493652571985554, -0.21148859722662888, -0.21121409190046114, 0.26454125260173966, 0.3053074971322147, 0.0715278017786787, 0.3520790923281247, 0.1923471853952217, 0.27518488716048134, 0.14674370062519204], 
                11: [-0.19103095264227782, -0.17446720587728148, -0.16012562878446304, -0.051615194420812444, -0.1918769902377767, -0.173932440690929, -0.19074791185378706, 0.05958963865957129, 0.08733763726932536, -0.12853099376165117, -0.21395974580569835], 
                12: [-0.1398512231497957, -0.08537906289997763, 0.39216525968032706, 0.25932046309196344, 0.2924403950175596, 0.24756022757737617, 0.01566090320088448, -0.1990834783760161, -0.12321198492529978, 0.007965577586252187, 0.07870331508850764, 0.2767772666757884], 
                13: [-0.06872738624508415, 0.024557652657053924, 0.2807233335947531, -0.056482011985700825, 0.0713573762345078, 0.3435777720314698, -0.18488608773511633, 0.372445055348846, -0.05346365056779112, -0.16877103642614852, -0.012629288813336415, 0.2175652604347121, 0.42016105222067324], 
                14: [0.29503577592677965, -0.04172482651093773, -0.1863902607106758, 0.061887345044428915, -0.2329121869197116, 0.1397180290385155, -0.1877167289558101, -0.15370237282702903, 0.26232336225298636, 0.2810670509643155, -0.19118166188032912, -0.17263048942722203, 0.38003860958059793, 0.39728765330389954], 
                15: [-0.14476550790995393, -0.10915286158415718, -0.13317156120259327, -0.11354669248872286, 0.3707046179972938, -0.16504412483986114, -0.16256869514325056, -0.1885091590913146, -0.001137425291739269, 0.08897818415343506, 0.3417129128735703, 0.3689073638933177, 0.3716530620037757, 0.46213303468240596, -0.2654209429862753], 
                16: [0.37918336074846765, 0.11563068083146905, 0.29525650201325826, -0.123703146117612, -0.10323216750558994, -0.05058142375103153, -0.19947048856389335, 0.36798444509395056, 0.3239250916857191, -0.18045056609404944, 0.355846095931298, -0.06559845244056908, 0.2772385717336108, -0.061460205567504894, 0.2978451489414572, 0.29793235643655736], 
                17: [0.11275438084462741, 0.1254725117663698, 0.24218982008404974, 0.36793180589611296, -0.1116931137590239, 0.3536853135106677, -0.17562558154372, 0.2690458497638517, 0.19278304848216996, -0.16984484771677752, 0.3619199737657409, 0.3292608465053978, -0.15896587424472342, 0.2641855720695554, 0.40361613054943224, -0.17610843829020945, 0.270408037155835], 
                18: [0.21417584339033688, 0.26591927790743897, -0.21802203693024075, -0.21282280010520407, -0.16403880197435566, -0.09158283022579541, -0.25028421421182234, -0.18616791745971106, 0.4377314847385635, -0.22520524805183562, -0.22730463289714156, -0.18938545342710786, 0.13859341894217023, 0.31833450491104637, 0.3237477160595409, 0.044678129201536576, 0.38401236586224197, -0.16057107030414006], 
                19: [-0.24584242944747547, 0.4193388084860853, -0.19402314322393732, 0.1291595769212796, 0.1438201824269622, -0.17409059104187, -0.18731274988171748, 0.1358089433563124, -0.18840146010937925, -0.33336409846480763, 0.27395343696375984, 0.11560243691684682, 0.32961648088611295, 0.37724530232230075, -0.2953600096170595, -0.20020511402563276, 0.14888362026019752, 0.401489429022753, 0.3285530607459426], 
                20: [0.44531965133833973, 0.3685398185215777, 0.22935897168916317, -0.08103653369314404, -0.23564848616110695, 0.20944041656284568, 0.31695765913406937, 0.2290463391849394, 0.17635819804696656, 0.038548396903273996, -0.16679766320437078, -0.3034666763199423, -0.21951888903549444, 0.0958075544736891, -0.23810652325138532, 0.25050653799792466, 0.337136085833249, -0.1391739448579561, 0.10806995589126478, 0.10569487906159708]
            }

            vae_upper_bounds = {
                1: [0.4659291264348312], 
                2: [0.20806672103340595, 0.6644120846492507], 
                3: [0.13498618500046564, 0.6667806461013062, 0.035824704322950915], 
                4: [0.07074987483413026, 0.09892980471459445, 0.5678005738576886, 0.6481047200662879], 
                5: [0.6176715777743339, 0.4404340777768916, 0.2315092305854225, 0.6520140024414781, 0.5841967954637194], 
                6: [0.09404133307090981, 0.08378572398130335, 0.08634122057356088, 0.12758768225884798, 0.5793750785552362, 0.540614262088701], 
                7: [0.3157099948319179, 0.09872400001972845, 0.6790926282552383, 0.6831477346977883, 0.7399263818868309, 0.1083001901223451, 0.5255823012302726], 
                8: [0.5232633465701189, -0.02889226109102748, 0.6059736828020663, 0.05808789333397117, 0.4916095529172875, 0.3461673281109935, 0.6570175686583688, 0.5305485461112558], 
                9: [0.7720941909041237, 0.09425011918993405, 0.7453960254555476, 0.4050745089571745, 0.1106939878814652, 0.5457860469601583, 0.5689446263194928, 0.46520177839376137, 0.5082383794146547], 
                10: [0.0592758058130904, 0.06847827119040888, 0.036057351673675825, 0.5983536758709802, 0.6661057232220144, 0.5357169935416045, 0.6383960202347184, 0.5441208522982359, 0.5227857501248198, 0.6663255196906563], 
                11: [0.2838599675894086, 0.11191806045817228, 0.046073251681996004, 0.5452250879299712, -0.00781331310126085, 0.14526774136325657, 0.26167171503081266, 0.7255011259758186, 0.44187852098437475, 0.41185658313371376, 0.11233728151295133], 
                12: [0.3286066276111974, 0.38859731538802045, 0.6588034031676251, 0.6362248183292813, 0.591315257810584, 0.6122804766289733, 0.6285033255820888, 0.01926356801208859, -0.04342001190343691, 0.3101873236093229, 0.3760921098859988, 0.5126940079057652], 
                13: [0.36652986228738504, 0.5298941720656822, 0.6419556597933247, 0.24692227866054553, 0.5107950581375098, 0.7769468380025703, 0.03352137892474913, 0.7729535885157643, 0.4677850347820065, 0.0827202534383232, 0.3843010874994034, 0.5799301563652384, 0.6403592222067114], 
                14: [0.5998121995924048, 0.2604352516698726, 0.11540031364148094, 0.5124080667341474, 0.14660129254717444, 0.6048757445938326, 0.0497483481702022, -0.04894771801643655, 0.6039421465181067, 0.5735365393905447, -0.04403427807842954, 0.17955281556411945, 0.7328145569225777, 0.6306636477973809], 
                15: [-0.013107212135474368, 0.34596020239025205, 0.33956765378867265, 0.018084758643724053, 0.6543582290685067, 0.07509807357205718, 0.08093973501422716, 0.057462444003872945, 0.6336715277490118, 0.8199806950500589, 0.7022493320011785, 0.770333280472986, 0.64065963144718, 0.7373880528651255, 0.1367292647311253], 
                16: [0.6205024319530875, 0.5539123870441468, 0.5952237697123024, 0.36188399847004843, 0.63798236022571, 0.6044334404085421, 0.13048305482865447, 0.6960058977592903, 0.5837880550428802, 0.07134363802506341, 0.6997401159204276, 0.41301832408639216, 0.5750650617461689, 0.3243921692775714, 0.5379188903968875, 0.6334929588974774], 
                17: [0.6887561806175302, 0.7358676918181011, 0.5178775312328697, 0.6282741751698034, 0.016146542854979087, 0.7456797632370086, 0.5452579992940475, 0.6879609696884671, 0.5872456276626377, 0.03263059282125892, 0.6756908777469118, 0.6190678609736684, 0.18798715683171216, 0.5939385955840759, 0.6642974296644795, 0.027514758903756403, 0.6894491789194224], 
                18: [0.6984693412557337, 0.6687080282649183, 0.11757730856752989, 0.06564504921343385, 0.013264804188515, 0.37874673683116267, 0.015603281821527482, 0.05780155231188447, 0.7063998453900429, 0.14928037581255382, 0.05298610509974751, 0.13334921575001302, 0.5309491808104274, 0.7792118771236951, 0.6884283723798591, 0.5767764707244212, 0.6395060673534142, 0.5196309602483399], 
                19: [0.22800033655743, 0.6536971693995504, 0.16598295113559106, 0.4462295562124358, 0.5666389430677008, 0.03215621601634615, 0.08965672860006062, 0.5971626989707712, 0.11170615731260373, 0.23322026615844668, 0.5292241694024153, 0.6010513855417123, 0.6842035580002922, 0.6331491767435614, 0.23693036135952497, 0.11834063200585612, 0.47645492232474734, 0.6376521229485237, 0.7165434888496967], 
                20: [0.7399333642607445, 0.5868396050436282, 0.564862370346203, 0.5022887574589014, 0.01676576597504366, 0.5503523748929458, 0.5873337877950314, 0.6307594317511753, 0.4476784169927416, 0.6369284323416116, 0.11952464762698621, 0.24347135404388312, 0.1587383416151963, 0.4310443091578884, 0.1813850637083994, 0.5786615244670017, 0.5408396127712842, 0.40421357677565284, 0.715436871994491, 0.668962046076688]
            }
            
            if encoder_model == "OTNAE":
                lower_bounds = [-0.14415649186087193, -0.268543757148559, -0.15896929669214185, -0.1374836113201868, -0.1231138271502097, -0.028467935630476518, -0.15394623407031857, -0.14735976794621808, -0.20080897097428035, 0.16406875734636858, -0.10864805966148694, 0.08775614397042864, -0.06409422315714391, -0.07623202192450722, 0.058931750632435595, -0.09724827970291301, -0.12364091842782947, 0.14362142735290517, -0.06895174363858043, 0.04642991179065827]
                upper_bounds = [0.26747567145435547, 0.14806290344345446, 0.17415004889215693, 0.08059228852219971, 0.15800166738648, 0.15360367453731355, 0.03721243806943259, -0.0039348626974199385, -0.06877675622499832, 0.26142192665734504, 0.011327333489360679, 0.1561538818892893, -0.007057088326002648, -0.0164353867111246, 0.11266980655988885, -0.051691908461773114, -0.08493031270336736, 0.17677282766451888, -0.03870521837687034, 0.06687843432872367]
            elif encoder_model == "AE":
                lower_bounds = ae_lower_bounds[latent_dim]
                upper_bounds = ae_upper_bounds[latent_dim]
            elif encoder_model == "VAE":
                lower_bounds = vae_lower_bounds[latent_dim]
                upper_bounds = vae_upper_bounds[latent_dim]
            elif encoder_model == "OTNVAE":
                lower_bounds = [-0.19626695428977312, -0.36711380360795637, 0.04259556123141203, -0.8425830113940911, -0.2020335318345339, 0.0643123292615412, -0.601876610763036, -0.5222595360014098, -0.38780405468000245, -0.13597708910264783, -0.4326257491754365, -0.018633573804109738, 0.3417419002171947, -0.4792846010327671, 0.11483036452574666, -0.11620678646161749, -0.10425779027010634, 0.3623712771788711, -1.0, -0.4455793527061583]
                upper_bounds = [0.3173220987470979, 0.005153532353760093, 0.34424266531809933, -0.49259944642972553, 0.13503396381913613, 0.3673335251135764, -0.19319593788207085, -0.3186197678636759, -0.1023573696667553, 0.14761019551978868, 0.05188481531174732, 0.18340628675982243, 0.9079186150559496, -0.3172629925866336, 0.4910767004460411, 0.23488229692619816, 0.26007389109219253, 0.8565963679978557, 0.06603000174023366, 1.0]

        if self.encoder_model != None:
            self.init_ae_model(native_dim, latent_dim, hidden_layer,model_path)
            # self.action_space = gym.spaces.box.Box(np.array(lower_bounds[:latent_dim]), np.array(upper_bounds[:latent_dim]), shape=(latent_dim,))


        self.num_links = num_links
        self.goal = np.array(goal)
        self.randomize_goal = goal is None
        self.step_number = 0

        self.goal_init = np.array([0,0]) # some bogus goal for initialization

        # if slope is None:
        #     slope = [-0.1, +0.1]
        # if gravity is None:
        #     gravity = [0.33, 3]
        # if friction is None:
        #     friction = [0.5, 1.5]
        # if density is None:
        #     density = [4000, 4000]

        # if mass is None:
            # mass = [0.5,2]
        # if damping is None:
        #     damping = [5, 5]


        self.max_norm = 0
        utils.EzPickle.__init__(self)
        MujocoEnv.__init__(self, model_path=f"reacher_{num_links}dof.xml", frame_skip=2,
                           slope_bounds=slope,
                           gravity_bounds=gravity,
                           mass_bounds=mass,
                           friction_bounds=friction,
                           density_bounds=density,
                           damping_bounds=damping
                           )

    def _step_goal(self):
        t = self.step_number
        center = np.array([0.7, 0.4]) * self.num_links/10
        rx = 0.1 * self.num_links/10
        ry = 0.2 * self.num_links/10

        x = rx * np.cos(2*np.pi/200 * t) + center[0]
        y = ry * np.sin(2*np.pi/200 * t) + center[1]
        self.goal = np.array([x, y])
        self.sim.data.qpos[-2:] = self.goal
        self.sim.data.qvel[-2:] = np.zeros(2)

        # norm = np.linalg.norm(self.goal)
        # if norm > self.max_norm:
        #     self.max_norm = norm
        #     print(self.max_norm)

    def step(self, a):
        # a = self.reconstruct(a)
        a = self.bottleneck(a, self.latent_dim)
        vec = self.get_body_com("fingertip") - self.get_body_com("target")
        # reward_dist = np.exp(-np.linalg.norm(vec))
        # reward_ctrl = 0
        reward_dist = -np.linalg.norm(vec) * self.num_links
        reward_ctrl = -np.square(a).sum()
        # print(a)
        # print(reward_ctrl, reward_dist, sep='\t')
        reward = reward_dist + reward_ctrl
        self.do_simulation(a, self.frame_skip)
        ob = self._get_obs()
        done = False # I think the max_steps in teh registration handles the horizon.

        info = dict(reward_dist=reward_dist, reward_ctrl=reward_ctrl)
        # print(info)

        self.step_number += 1
        self._step_goal()

        return ob, reward, done, info

    def viewer_setup(self):
        self.viewer.cam.trackbodyid = -1 # id of the body to track ()
        self.viewer.cam.distance = self.model.stat.extent * 0.4  # how much you "zoom in", model.stat.extent is the max limits of the arena
        self.viewer.cam.lookat[0] += 0.0  # x,y,z offset from the object (works if trackbodyid=-1)
        self.viewer.cam.lookat[1] += 0.0
        self.viewer.cam.lookat[2] += 0.0
        self.viewer.cam.elevation = -90  # camera rotation around the axis in the plane going through the frame origin (if 0 you just see a line)
        self.viewer.cam.azimuth = 0  # camera rotation around the camera's vertical axis

    def reset_model(self):
        self.step_number = 0

        qpos = (
            self.np_random.uniform(low=-0.2, high=0.2, size=self.model.nq)
            + self.init_qpos
        )

        # reset goal
        self.goal = self.goal_init
        qpos[-2:] = self.goal

        qvel = self.init_qvel + self.np_random.uniform(
            low=-0.005, high=0.005, size=self.model.nv
        )
        qvel[-2:] = 0
        self.set_state(qpos, qvel)
        return self._get_obs()

    def _get_obs(self):
        theta = self.sim.data.qpos.flat[:2]
        return np.concatenate(
            [
                np.cos(theta),
                np.sin(theta),
                self.sim.data.qpos.flat[self.num_links:],
                self.sim.data.qvel.flat[:self.num_links],
                self.get_body_com("fingertip") - self.get_body_com("target"),
            ]
        )