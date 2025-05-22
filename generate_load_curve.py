#%% Creation de la fonction
# Script Python pour créer une load curve pour FEBio avec fonction personnalisée
def generate_load_curve(file_path, func, start=0, end=40, step=1):
    """
    file_path : chemin vers le fichier de sortie
    func      : fonction Python prenant x en argument, ex: lambda x: x + 1
    start     : valeur de départ pour x
    end       : valeur finale pour x
    step      : pas d'incrémentation
    """
    with open(file_path, 'w') as f:
        for x in range(start, end+1, step):
            y = func(x)
            f.write(f"{x} {y}\n")

#%% test
# Exemple d'utilisation avec f(x) = x/40 + 0
generate_load_curve("C:/Users/jr403s/Documents/Test_segmentation_itk/Python_vtk_Slices/dtmax.txt", func=lambda x: 1)

# %%
