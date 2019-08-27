import datetime

from colorama import Fore

import helper as hp
import preprocessing as pp
import bayesian as bn

if __name__ == '__main__':
    print( Fore.GREEN + f"-----------------------------------------" )
    print( f"{datetime.datetime.now()}" )
    print( f"-----------------------------------------" )

    df = hp.read_dataset( '../DBMedico/DBBCTI_20042014_VMMZ_GL.xls', "excel", sheet="DB3" )


    print( f"Number of lines before cleaning: {len( df )}" )

    # drop records according to specifications
    df = pp.drop_records( df )

    print( f"Number of lines after cleaning: {len( df )}" )

    # bin data in columns according to specifications
    df = pp.bin_records( df )

    # df = df[["ki67", "differenziazione", "recettori estrogeni"]]

    NUM_VALUES = len( df.columns )

    # make dictionaries mapping categorical codes to original values and viceversa
    df_values, df_codes, code_to_value_map, value_to_code_map = pp.make_mappings( df, NUM_VALUES )

    # slice original dataset into value and code datasets
    # (df_values, df_codes) = pp.slice_codes(df, NUM_VALUES)

    from bayesian import BN

    print( "Building Bayesian Network ..." )
    model = BN( df_codes, df_values, code_to_value_map, value_to_code_map )

    #############################################################

    # model.plot_model( independenciesplot=False, directed=True )
    #
    # var_source = "eta arrotondata"
    # var_evidence = ["mut17q21"]
    # separations = model.d_separated( var_source, var_evidence )
    # model.plot_model( independenciesplot=True, directed=True, evidence=var_evidence, separations=separations, source=var_source,
    #                   showconnectedness=False )

    evidence = model.initial_random_evidence()
    # evidence = {'differenziazione': 1, 'loss 17': 2, 'FISH': 3, 'c erbB 2': 1, 'ki67': 3, 'morfologia': 1, 'pM': 1}
    # evidence = {}

    mpe_pgmpy_ve = model.mpe_query( evidence, algorithm="Variable Elimination" )

    # MPE external solver
    path = "../export/"
    # model.export_model_to_uai( path + "my.uai", format="DAOOPT" )
    model.export_evidence_to_uai( path + "evidence.uai.evid", evidence, format="pgmpy" )

    from pgmpy.readwrite import UAIWriter, UAIReader

    writer = UAIWriter( model.model_pgmpy )
    writer.write_uai( path + "pgmpy.uai" )
    # reader = UAIReader( path + 'pgmpy.uai' )
    # model = reader.get_model()

    # mpe_daoopt_mine = bn.daoopt_solver( path + "my.uai" , path + "my.uai.evid", path + "sol_my.txt", model.node_names, evidence )
    mpe_daoopt_pgmpy = bn.daoopt_solver( path + "pgmpy.uai" , path + "evidence.uai.evid", path + "sol_pgmpy.txt", model.node_names, evidence, "pgmpy", model.model_pgmpy )

    print("pgmpy\n", sorted( mpe_pgmpy_ve.items(), key=lambda x: (x[0], x[1]) ))
    # print("DAOOPT my UAI", mpe_daoopt_mine)
    print("DAOOPT pgmpy UAI\n", sorted( mpe_daoopt_pgmpy.items(), key=lambda x: (x[0], x[1]) ))

