import gym
import numpy as np
from gym import utils
#from lxml import etree

from gym_mujoco.envs.mujoco_env import MujocoEnv

class SwimmerOldEnv(MujocoEnv, utils.EzPickle):
    def __init__(self, num_links=6,
                 reward_ctrl=True,
                 boost=False,
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
                model_path=None
    ):
        self.encoder_model = encoder_model
        self.native_dim = native_dim
        self.latent_dim = latent_dim
        self.ae_model = None

        if num_links == 6:
            ae_lower_bounds = {
                1: [-1.0], 
                2: [-0.7702582660054329, -1.0], 
                3: [-1.0, -1.0, -0.7804031507010476], 
                4: [-1.0, -0.7204855261233142, -0.42783486354263384, -0.880484662832257], 
                5: [-0.7848684069854572, -0.6850983104931286, -0.9163480372512285, -0.8533028011419663, -0.3947643902471676]
            }

            ae_upper_bounds = {
                1: [0.852891862346702], 
                2: [1.0, 1.0], 
                3: [0.9066548169002201, 1.0, 1.0], 
                4: [0.743500280261494, 0.5221095931819529, 0.4227319102246632, 1.0], 
                5: [0.6904387815323838, 0.7230809523935938, 1.0, 0.6642160685764625, 0.8719501737056636]
            }

            vae_lower_bounds = {
                1: [-0.6003595876234047], 
                2: [-0.06872315998444734, -0.49112376741697017], 
                3: [-0.1817924712482384, -0.4063452928503203, -0.3500295737099809], 
                4: [-0.5068289560212742, -0.7261314083579828, 0.11561353057115054, -0.3680128079364192], 
                5: [-0.7302718685541251, -0.616813373898808, -0.4224241584590478, -0.40812832766368284, 0.27783095056499063]
            }
            vae_upper_bounds = {
                1: [1.0], 
                2: [1.0, 0.9946192632853665], 
                3: [-0.03084287723191348, 1.0, 1.0], 
                4: [0.8871340451171623, 1.0, 0.9026499846606078, 1.0], 
                5: [1.0, 1.0, 0.6374102900937849, 1.0, 0.9607998215106495]
            }
            
            if encoder_model == "OTNAE":
                lower_bounds = [-1.0, -0.93843647408515, -0.23334034882502933, -0.3979220633910589, -0.015145718338627628]
                upper_bounds = [1.0, 0.7125006700985733, 0.1855612823459244, 0.3576527958262061, 0.2762315728949867]
            elif encoder_model == "AE":
                lower_bounds = ae_lower_bounds[latent_dim]
                upper_bounds = ae_upper_bounds[latent_dim]
            elif encoder_model == "VAE":
                lower_bounds = vae_lower_bounds[latent_dim]
                upper_bounds = vae_upper_bounds[latent_dim]
            elif encoder_model == "OTNVAE":
                lower_bounds = [-0.8716157712214319, -0.5210172396873902, -0.5110839178639635, -1.0, -1.0]
                upper_bounds = [0.9793826567396601, 0.22866516097483852, 0.10199056648900667, 1.0, 1.0] 

        if num_links == 10:
            ae_lower_bounds = {
                1: [-1.0], 
                2: [-1.0, -1.0], 
                3: [-1.0, -1.0, -1.0], 
                4: [-1.0, -1.0, -0.8028865108264518, -0.7022620885678452], 
                5: [-1.0, -0.7806748975058082, -0.405181620721781, -0.7506656969482552, -1.0], 
                6: [-1.0, -0.7670559629386314, -1.0, -0.761528412859254, -1.0, -1.0], 
                7: [-1.0, -1.0, -0.5498737064433659, -0.869135621762058, -1.0, -1.0, -0.9473165345728378], 
                8: [-0.6488154766594403, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -0.7687062993481187], 
                9: [-1.0, -1.0, -0.6245459935797278, -0.8657728723588645, -1.0, -1.0, -1.0, -0.7013502253058942, -0.794735246781272]
            }

            ae_upper_bounds = {
                1: [1.0], 
                2: [1.0, 1.0], 
                3: [1.0, 1.0, 0.837537933130807], 
                4: [1.0, 1.0, 1.0, 0.6969141107228852], 
                5: [1.0, 0.896358016134495, 0.5255643236952048, 0.9464678373933584, 1.0], 
                6: [1.0, 0.7632075560585604, 1.0, 0.7671517477790889, 1.0, 0.8630458945854427], 
                7: [1.0, 0.970155955210729, 0.5391233449786088, 0.6242613034949808, 0.8447930320062387, 1.0, 0.6373380413825828], 
                8: [0.4852963557126717, 1.0, 1.0, 1.0, 1.0, 0.9556426988693545, 1.0, 0.8588382839598273], 
                9: [1.0, 1.0, 1.0, 0.7077036707208163, 1.0, 1.0, 1.0, 0.8699479381141658, 0.5541452787998089]
            }

            vae_lower_bounds = {
                1: [-1.0], 
                2: [0.14890976810743217, -0.764644443734525], 
                3: [-0.010641023771127078, -0.6870117203042265, -0.47898678119465476], 
                4: [-0.4311222894188937, -0.8322034425228046, -0.58185350279514, -0.9937385838509278], 
                5: [-1.0, -0.38622460212702575, -0.5550679743937296, -0.534691489319849, -1.0], 
                6: [-0.8004438579760406, -0.7081920510145122, -0.12873398865679864, -0.45242472944881984, -0.7979940134643385, -0.5894467351687362], 
                7: [-1.0, -0.9503540136293918, -0.957802490856213, -0.7443519027102139, -1.0, -0.5674470019436526, -1.0], 
                8: [-1.0, -0.8839657983975204, -0.9175414159781163, -0.8766620402515765, -1.0, -0.413151166478745, -0.5384492529414628, -0.8303159227320636], 
                9: [-0.5998307092650431, -0.559165001956339, -0.32467998568061873, -1.0, -1.0, -0.9756704954378588, -0.7944589648793194, -0.5401609814606119, -0.8579459315827138]
            }
            vae_upper_bounds = {
                1: [1.0], 
                2: [0.7922381717084765, 1.0], 
                3: [0.6908567785595366, 0.9325845849940333, 1.0], 
                4: [1.0, 1.0, 1.0, 1.0], 
                5: [1.0, 0.25601672274209153, 0.5715385188162151, 1.0, 1.0], 
                6: [1.0, 1.0, 0.9816792773675388, 1.0, 1.0, 1.0], 
                7: [1.0, 1.0, 1.0, 1.0, 1.0, 0.7503364632395711, 1.0], 
                8: [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0], 
                9: [1.0, 1.0, 0.695805264000909, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
            }
            
            if encoder_model == "OTNAE":
                lower_bounds = [-1.0, -0.4902837233763428, -0.49346864222468306, -0.33319172215116377, -0.3598006814248238, -0.35414054252903454, -0.18394538829123125, -0.3732934668372093, -0.5744426332770299]
                upper_bounds = [1.0, 0.7546523616564245, 0.28092640381704503, 0.2996048877107843, 0.46812252608580074, 0.4411567861055503, 0.2937999550904452, 0.026147690760189968, 0.27111071665374176]
            elif encoder_model == "AE":
                lower_bounds = ae_lower_bounds[latent_dim]
                upper_bounds = ae_upper_bounds[latent_dim]
            elif encoder_model == "VAE":
                lower_bounds = vae_lower_bounds[latent_dim]
                upper_bounds = vae_upper_bounds[latent_dim]
            elif encoder_model == "OTNVAE":
                lower_bounds = [-1.0, -1.0, -0.7412832145391628, -0.53734508017438, -0.9954180864923708, -1.0, -1.0, -1.0, -1.0]
                upper_bounds = [1.0, 0.6340846507806454, 0.43906876544913187, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
        if num_links == 20:
            ae_lower_bounds = {
                1: [-1.0], 
                2: [-1.0, -1.0], 
                3: [-0.9309080517804614, -0.8470308135079372, -0.9760083969379231], 
                4: [-0.8637530643021205, -0.9420054008550911, -0.8787267379479429, -0.9187227635574621], 
                5: [-0.8257986152848689, -0.6787008096573376, -1.0, -1.0, -0.7441786198078045], 
                6: [-0.7539926168852544, -0.6729033908513113, -0.729005972639486, -0.8998975799762239, -1.0, -1.0], 
                7: [-0.7086591725378657, -0.7004799185987981, -0.6965043235064233, -1.0, -0.7543282095545051, -0.9704869438479439, -0.6544125222959923], 
                8: [-0.7280844334180281, -0.7080247505375797, -1.0, -0.7636916100140346, -0.7495826128932919, -0.7866755481656191, -0.763826251890619, -1.0], 
                9: [-0.8507984249494815, -0.8876896215148433, -0.9161052421149696, -0.7382437393863113, -0.8480230046810362, -0.8711832075087794, -0.8136053499410502, -0.7018384942968132, -0.7595956397958031], 
                10: [-0.9494742380194466, -0.787599522939497, -0.6000177188285613, -0.6593892850097233, -1.0, -0.6929940832325133, -0.7253097751899384, -0.7091990168806199, -0.8036500661864623, -1.0], 
                11: [-0.8750463231884138, -0.8475370464133852, -0.6101698107698562, -0.6113487594487151, -0.792699195040816, -0.9232347237900492, -0.706839223226565, -0.727824003491657, -1.0, -1.0, -0.7486308382148156], 
                12: [-0.868898711425391, -1.0, -0.8513295107302387, -0.9858860819243106, -0.9229600906276998, -0.8222884128133261, -0.8587222571977039, -0.800960425985406, -0.6978872850740231, -0.7395693096515424, -1.0, -0.9504934260336716], 
                13: [-1.0, -0.9162725071757143, -0.8666349028709137, -0.730602654709853, -1.0, -0.7674483572104248, -0.7770705342323968, -0.7499535740141784, -0.6940643441418555, -0.9572386386430229, -0.5382469847694564, -0.6255076496348516, -0.978599819775852], 
                14: [-0.7524479369152078, -0.7073618361050581, -0.8144718226108001, -0.7313419238127286, -0.832214509131907, -0.7508378427325018, -0.8301841450868175, -0.6420002956073605, -0.8868639430496866, -0.8151760784439886, -0.7528845343662582, -1.0, -0.8422218846321405, -0.8013847138649713], 
                15: [-1.0, -1.0, -1.0, -0.643019801396242, -0.7700217940962791, -0.8681289262438746, -1.0, -0.5553846800091622, -0.8499686451092034, -0.8399197412472087, -0.7135473178018537, -0.6227043727570114, -0.5973739285021239, -0.9116879050253021, -0.7611046150359561], 
                16: [-0.7467487126661391, -0.7037963376505254, -0.5921513122423022, -0.64655992703863, -0.6528530869549776, -0.6915984320140673, -0.1286799321684549, -1.0, -0.5831662857729889, -1.0, -0.6518841432306622, -1.0, -0.7196727863129824, -0.5005280118863596, -1.0, -0.7651509234311988], 
                17: [-0.9873213427099342, -1.0, -0.9128976646819091, -0.5867846233037712, -1.0, -0.7741017497348733, -0.8873643982713005, -1.0, -0.5746427553822966, -1.0, -0.6475713861189898, -0.7357064636253715, -0.832539912910851, -0.7813240237789361, -1.0, -0.8407637326790695, -0.7133468031148014], 
                18: [-0.9570134417212859, -0.7513487656804179, -0.9810331615965884, -0.5781548241329602, -0.7732201262793299, -0.7462869178896064, -1.0, -1.0, -0.7817002123257422, -0.9489511110301846, -0.4871782517493165, -1.0, -0.6802991668524146, -0.4876211749919922, -1.0, -0.9861558200467839, -0.6339876694385321, -1.0], 
                19: [-0.7000232067164348, -1.0, -0.5171677948370954, -0.7639468489951003, -0.6306501363127669, -0.8297714140257823, -1.0, -1.0, -0.6313486373932911, -0.7335683305793255, -0.7391255042003341, -0.7549399098527526, -0.9254325169344824, -0.8491321555806564, -1.0, -0.6793759984389632, -0.936814226595603, -0.5235587337154943, -0.5034957604179371]
            }

            ae_upper_bounds = {
                1: [1.0], 
                2: [0.9660024021011001, 1.0], 
                3: [0.9145963192654852, 0.8421361902759129, 0.9164648179452757], 
                4: [0.8824619383148479, 0.8325990860187192, 0.8462387157905708, 0.9280956502170736], 
                5: [0.9102386507937432, 0.6615229778444405, 0.994928696975792, 0.9424918910860594, 0.7965136730989517], 
                6: [0.7085438914596149, 0.6270464433585518, 0.6052675645658251, 0.8153886773898552, 0.9947325249583255, 1.0], 
                7: [0.8951210137951557, 0.640399432988995, 0.8638201013749764, 1.0, 0.9015631189236333, 0.8258969015293355, 0.6781379255631663], 
                8: [0.71709860567133, 0.7028173207855096, 1.0, 0.6690129927121694, 0.7316060106272207, 0.5713896677525014, 0.7389031922556784, 1.0], 
                9: [0.7713036239751969, 0.9153196343388329, 0.75098735464991, 0.8927727576337459, 0.7979908563983165, 0.8973497211150463, 0.6398364521688783, 0.7518763369024007, 0.6479004558121975], 
                10: [0.8773599030584618, 0.7505288124940026, 0.7797688401103318, 0.6546036937278449, 1.0, 0.7594841852175442, 0.7366497265346402, 0.7902702812243795, 0.659321443553699, 1.0], 
                11: [0.8687446844463942, 0.9263660173937085, 0.8447584539603543, 0.7585610562004951, 0.7744830006876765, 0.6204135759661841, 0.740341629464867, 0.5353901112389611, 1.0, 1.0, 0.6059190509823834], 
                12: [0.8827193278488817, 1.0, 0.7716884441505713, 1.0, 0.7252689925994119, 0.7995558210115294, 0.9337111435264256, 0.8450859984491066, 0.6629475000419248, 0.7761134778445025, 1.0, 1.0], 
                13: [1.0, 1.0, 0.9215369955590544, 0.6382558448413854, 1.0, 0.6431068731674944, 0.6232419140837544, 0.4516955759145701, 0.5804788193336289, 0.8874014754380466, 0.6883521636797465, 0.5326744626287743, 0.7984288360994634], 
                14: [0.7283155845914346, 0.8293362400819464, 0.6139638236156034, 0.5949397262521184, 0.7923407424800221, 0.6370940659746037, 0.5211185562516508, 0.6392558040537758, 0.9970874455132266, 1.0, 0.8836278167279574, 1.0, 0.5625128705106399, 1.0], 
                15: [0.5695092401298408, 1.0, 1.0, 0.8467871499366539, 0.7403143030337868, 0.7650863733858586, 1.0, 0.7622482440223697, 0.8315813024467132, 0.881510481439096, 0.6509302845392748, 0.8098127973471846, 0.8284752171006802, 0.711610963229564, 0.817356494218554], 
                16: [0.5511118275814092, 0.6915581568140321, 0.6883465179203004, 0.7070013365554808, 0.8115364896526659, 0.5817612295643709, 1.0, 0.9862266490313335, 0.7266061069632351, 1.0, 0.7033149288687864, 1.0, 0.8308093078959184, 0.702620121610356, 0.0005810380450439467, 0.44254693665632394], 
                17: [1.0, -0.21408393868655506, 0.7537094257156429, 0.7588814516106074, 1.0, 0.8165103479842162, 0.8051394990379755, 1.0, 0.6556260632574057, 1.0, 0.7663932867714873, 0.71435068315078, 0.7012352971317798, 0.7159068113976746, 1.0, 0.45818294988892055, 0.7294023014500033], 
                18: [0.945557168889654, 0.6982111173245036, 0.5953793849257825, 0.7134427688961611, 0.6209155715953186, 0.7174674887516166, 1.0, 0.4397341530026504, 0.6485981852335774, 0.8785338715108509, 1.0, 1.0, 0.6470171158090701, 1.0, 0.8553285905563607, 0.7257186732309047, 0.874023853703298, 1.0], 
                19: [0.8818699679270164, 1.0, 0.9874078103277821, 0.7158261459794923, 0.919549366732689, 0.7906953324378718, 0.09681996890731442, 1.0, 0.714148121232644, 0.5325329220753459, 0.5911097588674329, 0.880151936787411, 1.0, 0.9228470696373401, 1.0, 0.6353190917924089, 0.5620298232125917, 1.0, 0.7919565617982636]
            }

            vae_lower_bounds = {
                1: [-0.7630337366537843], 
                2: [-0.9510775656872674, -0.9967983749203813], 
                3: [-1.0, -0.9932165876991272, -0.7582175952000776], 
                4: [-0.5614616741079665, -0.8256005190390607, -0.819855211109103, -0.8020048813048147], 
                5: [-0.5485788008812906, -0.7723261839991785, -0.5270499691934578, -0.7805475071246514, -0.6553801261271195], 
                6: [-1.0, -0.899268507216337, -0.8360565978411065, -1.0, -0.7957390564565385, -0.7185827759097377], 
                7: [-0.4499679809837114, -0.4832056482990666, -0.9947480571528031, -0.6474741473083016, -0.6778465995691232, -0.44200711134893866, -0.6015645895429181], 
                8: [-0.9156267899082678, -0.6358145662474821, -0.75321397714536, -0.6985243925233975, -0.8736966786511022, -0.7964182195383432, -1.0, -0.6038259931215123], 
                9: [-0.42644893124186534, -0.5789541222637422, -0.7396458261718224, -0.5916955278475624, -1.0, -1.0, -1.0, -1.0, -0.6937951099358874], 
                10: [-0.9672184966100028, -1.0, -1.0, -1.0, -0.4686027998092387, -0.8962419175421323, -0.6981510700953082, -0.7800208723771536, -0.5389043153645551, -0.8477388575531501], 
                11: [-0.2590263625794383, -0.7187892233420794, -0.5996353257180675, -1.0, -0.7784764354083227, -0.6371901008552106, -1.0, -0.8609373799025801, -0.5321795429438788, -1.0, -0.3384779267886351], 
                12: [-0.4249662638221562, -0.4182116438103653, -0.7662809489148625, -1.0, -0.26656966820706507, -0.8087601294030815, -0.7459563360824164, -1.0, -0.9094188349733254, -0.6396917909892832, -0.5739268176143675, -0.9807313144974432], 
                13: [-0.43046075182047827, -0.9219431143695243, -1.0, -0.868063546087128, -1.0, -0.566049040715797, -0.764073252710366, -0.8959871028666924, -0.4087428859482437, -0.544232585428872, -0.7767473334180416, -0.43718218795929387, -1.0], 
                14: [-0.5700181710412415, -0.5952162664116063, -0.824416629131602, -0.5395284498508361, -0.6384326970849543, -1.0, -0.910094819494082, -0.6384345774603328, -0.33718042244327945, -0.9488455203112147, -0.7371398858858067, -1.0, -0.3934204590551966, -1.0], 
                15: [-0.4047823751922653, -0.8875172732868495, -0.8741590417514031, -0.6337659243598319, -1.0, -0.668528650978703, -0.512385110509515, -1.0, -0.8141896308000853, -0.5242010897697762, -0.7827772824727547, -0.5196406760133416, -0.6579202552579494, -0.8014996424971157, -0.5426493028611497], 
                16: [-0.6584405958862467, -1.0, -0.5270181567738477, -0.20266153152469657, -1.0, -0.542136858490006, -1.0, -0.480040637494348, -1.0, -0.5177047668945638, -1.0, -0.849906231626732, -0.9839380403215436, -0.8961898231370528, -0.4318011256499675, -0.8966298113276663], 
                17: [-1.0, -0.9059273829274919, -1.0, -0.5730598291913792, -0.9850632546213243, -0.7496622339972532, -0.6371986541232557, -1.0, -1.0, -1.0, -0.7235954784696819, -0.8154585682727479, -0.935953708439188, -0.9560849822559867, -0.785822451947785, -0.6637837492471264, -0.8339203387698199], 
                18: [-0.8041220603964805, -1.0, -1.0, -0.5069587904615888, -0.775752419463602, -1.0, -1.0, -0.9780755595027096, -0.8525053610240132, -0.2608536531323251, -0.7204071936506671, -0.6017715679219714, -1.0, -0.607820000864707, -0.8964337020341242, -0.6343602172683473, -0.93425957263174, -0.44452666353723025], 
                19: [-0.6171955966588951, -1.0, -1.0, -0.48298622065928354, -0.889088861174241, -0.945071042719267, -0.8090601459375856, -0.44112406893284783, -0.7006302091033849, -0.4073548836899528, -0.48209077730396455, -1.0, -0.6746235842676633, -0.53679985711592, -0.6803150884929248, -1.0, -0.7137945562632809, -0.9740515360042555, -0.855463938686992]
            }
            vae_upper_bounds = {
                1: [1.0], 
                2: [1.0, 1.0], 
                3: [1.0, 1.0, 1.0], 
                4: [0.4354572369801451, 1.0, 1.0, 1.0], 
                5: [0.9723100049548575, 1.0, 1.0, 1.0, 1.0], 
                6: [1.0, 1.0, 1.0, 1.0, 1.0, 1.0], 
                7: [1.0, 1.0, 1.0, 0.8130774271485348, 1.0, 0.9291512733254017, 0.6534026599731616], 
                8: [1.0, 1.0, 1.0, 1.0, 1.0, 0.9906079830821216, 1.0, 0.7408844868587561], 
                9: [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0], 
                10: [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0], 
                11: [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0], 
                12: [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0], 
                13: [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0], 
                14: [1.0, 0.9868667602506144, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0], 
                15: [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.7487468969394465], 
                16: [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0], 
                17: [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0], 
                18: [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.7004780972079505], 
                19: [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
            }
            
            if encoder_model == "OTNAE":
                lower_bounds = [-1.0, -1.0, -1.0, -1.0, -0.5747713346472688, -0.6032289774509673, -0.6625612626891384, -0.6250290962387085, -0.4471479876272471, -0.667640165806469, -0.5134514380541524, -0.6931264174248077, -0.5318587839023439, -0.6996898840973754, -0.35350758575421615, -0.7816637279985138, -0.4757471126319585, -0.4952163318807645, -0.5368280240371037]
                upper_bounds = [1.0, 1.0, 1.0, 1.0, 0.6246427165711042, 0.46585349958216515, 0.4602697297752829, 0.4322053351655726, 0.5325957811933023, 0.4084149041231247, 0.40504405234136914, 0.5064478679181685, 0.6606963937751091, 0.4070739874444177, 0.7217553743978966, 0.7668188131379707, 0.5199730082880754, 0.3298845390257596, 0.40337509621261336]
            elif encoder_model == "AE":
                lower_bounds = ae_lower_bounds[latent_dim]
                upper_bounds = ae_upper_bounds[latent_dim]
            elif encoder_model == "VAE":
                lower_bounds = vae_lower_bounds[latent_dim]
                upper_bounds = vae_upper_bounds[latent_dim]
            elif encoder_model == "OTNVAE":
                lower_bounds = [-0.9152125673703071, -0.8395708911538196, -1.0, -1.0, -0.8625825060823349, -0.7225466055364841, -0.627292043533634, -0.45280984620699716, -0.8232512947797304, -0.6946435426168343, -0.7625907976904902, -0.9031043968689747, -0.7762308670622569, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0]
                upper_bounds = [0.932363257475678, 0.9435214068683622, 1.0, 1.0, 0.8270370769691536, 0.5852419476535191, 0.47228741735913327, 0.7939017429674595, 0.5727881984254412, 0.9990137872754121, 0.8734863041724036, 0.852421029172518, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
        
        if self.encoder_model != None:
            self.init_ae_model(native_dim, latent_dim, hidden_layer,model_path)
            # self.action_space = gym.spaces.box.Box(np.array(lower_bounds[:latent_dim]), np.array(upper_bounds[:latent_dim]), shape=(latent_dim,))

        self.num_links = num_links
        self.boost = boost
        self.reward_ctrl = reward_ctrl

        # if randomize_slope:
        #     slope_bounds = [-0.1, +0.1]
        # if randomize_gravity:
        #     gravity_bounds = [0.33, 3]
        # if randomize_mass:
        #     mass_bounds = [0.8, 1.2]
        # if randomize_friction:
        #     friction_bounds = [0.5, 1.5]
        # if randomize_density:
        #     density_bounds = [2000, 6000]
        # if randomize_damping:
        #     damping_bounds = [0, 5]

        MujocoEnv.__init__(self, f"swimmer{num_links}.xml", 4,
                           slope_bounds=slope,
                           gravity_bounds=gravity,
                           mass_bounds=mass,
                           friction_bounds=friction,
                           density_bounds=density,
                           damping_bounds=damping
                           )
        utils.EzPickle.__init__(self)

    def step(self, a):
        # a = self.reconstruct(a)
        a = self.bottleneck(a, self.latent_dim)
        ctrl_cost_coeff = 0.0001
        xposbefore = self.sim.data.qpos[0]
        self.do_simulation(a, self.frame_skip)
        xposafter = self.sim.data.qpos[0]
        reward_fwd = (xposafter - xposbefore) / self.dt
        reward_ctrl = 0
        if self.reward_ctrl:
            reward_ctrl = -ctrl_cost_coeff * np.square(a).sum()

        reward = reward_fwd + reward_ctrl
        ob = self._get_obs()
        info =dict(reward_fwd=reward_fwd, reward_ctrl=reward_ctrl)
        # print(info)
        return ob, reward, False, info

    def _get_obs(self):
        qpos = self.sim.data.qpos
        qvel = self.sim.data.qvel
        return np.concatenate([qpos.flat[2:], qvel.flat])

    def reset_model(self):
        vel = self.init_qvel + self.np_random.uniform(low=-0.01, high=0.01, size=self.model.nv)
        if self.boost:
            vel[0] = 1
        self.set_state(
            self.init_qpos
            + self.np_random.uniform(low=-0.01, high=0.01, size=self.model.nq),
            vel,
        )
        return self._get_obs()

    def viewer_setup(self):
        self.viewer.cam.trackbodyid = 0 # id of the body to track ()
        self.viewer.cam.distance = self.model.stat.extent * 0.3  # how much you "zoom in", model.stat.extent is the max limits of the arena
        self.viewer.cam.lookat[0] += 10.0  # x,y,z offset from the object (works if trackbodyid=-1)
        self.viewer.cam.lookat[1] += 0.0
        self.viewer.cam.lookat[2] += 0.0
        self.viewer.cam.elevation = -20  # camera rotation around the axis in the plane going through the frame origin (if 0 you just see a line)
        self.viewer.cam.azimuth = 160  # camera rotation around the camera's vertical axis