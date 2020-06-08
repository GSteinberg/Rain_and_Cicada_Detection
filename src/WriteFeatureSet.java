import org.apache.commons.math3.complex.Complex;

import java.io.*;

/**
 * Created by aalle on 25/07/2017.
 */
public class WriteFeatureSet {
    public static void main(String[] args)
    {
        boolean hasHPF=false;					//Set to true for 1 kHz high-pass filtered audio
        boolean isDownsampled=true;				//Set to true if sample rate=22.5 kHz (used in our testing), false assumes sample rate of 44.1 kHz
        File[] toFilter=getFilesInDirectoryWithExtension("/Path/To/Files",".wav");
        try
        {
            BufferedWriter writer = new BufferedWriter(new FileWriter(new File("/Path/To/Featuers/Features.arff"), false));
            writer.write("@relation Indices");
            writer.newLine();
            writer.write("@attribute fileName STRING");
            writer.newLine();
            writer.write("@attribute psd NUMERIC");
            writer.newLine();
            writer.write("@attribute snr NUMERIC");
            writer.newLine();
            writer.write("@attribute BedoyaSNR NUMERIC");
            writer.newLine();
            writer.write("@attribute SegSNR NUMERIC");
            writer.newLine();
            writer.write("@attribute spectralEntropy NUMERIC");
            writer.newLine();
            writer.write("@attribute temporalEntropy NUMERIC");
            writer.newLine();
            writer.write("@attribute bgn NUMERIC");
            writer.newLine();
            writer.write("@attribute stDevBgn NUMERIC");
            writer.newLine();
            writer.write("@attribute aci NUMERIC");
            writer.newLine();
            writer.write("@attribute lowSpectralCover NUMERIC");
            writer.newLine();
            writer.write("@attribute highSpectralCover NUMERIC");
            writer.newLine();
            if(!hasHPF)
            {
                for(int i=0;i<2;i++) {
                    writer.write("@attribute psd" + (i * 500) + " NUMERIC");
                    writer.newLine();
                    writer.write("@attribute snr" + (i * 500) + " NUMERIC");
                    writer.newLine();
                    writer.write("@attribute BedoyaSNR"+(i*500)+" NUMERIC");
                    writer.newLine();
                    writer.write("@attribute SegSNR"+(i*500)+" NUMERIC");
                    writer.newLine();
                    writer.write("@attribute spectralEntropy" + (i * 500) + " NUMERIC");
                    writer.newLine();
                    writer.write("@attribute aci" + (i * 500) + " NUMERIC");
                    writer.newLine();
                    writer.write("@attribute lowSpectralCover" + (i * 500) + " NUMERIC");
                    writer.newLine();
                    writer.write("@attribute highSpectralCover" + (i * 500) + " NUMERIC");
                    writer.newLine();
                }
            }
            for(int i=0;i<5;i++) //psdSubset[k]+","+snrSubset[k]+","+specEntropySubset[k]+","+aciSubset[k]
            {
                writer.write("@attribute psd"+(i*2000+1000)+" NUMERIC");
                writer.newLine();
                writer.write("@attribute snr"+(i*2000+1000)+" NUMERIC");
                writer.newLine();
                writer.write("@attribute BedoyaSNR"+(i*2000+1000)+" NUMERIC");
                writer.newLine();
                writer.write("@attribute SegSNR"+(i*2000+1000)+" NUMERIC");
                writer.newLine();
                writer.write("@attribute spectralEntropy"+(i*2000+1000)+" NUMERIC");
                writer.newLine();
                writer.write("@attribute aci"+(i*2000+1000)+" NUMERIC");
                writer.newLine();
                writer.write("@attribute lowSpectralCover"+(i*2000+1000)+" NUMERIC");
                writer.newLine();
                writer.write("@attribute highSpectralCover"+(i*2000+1000)+" NUMERIC");
                writer.newLine();
            }
            if(!isDownsampled)
            {
                    writer.write("@attribute psd" + 11000 + " NUMERIC");
                    writer.newLine();
                    writer.write("@attribute snr" + 11000 + " NUMERIC");
                    writer.newLine();
                    writer.write("@attribute BedoyaSNR"+11000+" NUMERIC");
                    writer.newLine();
                    writer.write("@attribute SegSNR"+11000+" NUMERIC");
                    writer.newLine();
                    writer.write("@attribute spectralEntropy" + 11000 + " NUMERIC");
                    writer.newLine();
                    writer.write("@attribute aci" + 11000 + " NUMERIC");
                    writer.newLine();
                    writer.write("@attribute lowSpectralCover" + 11000 + " NUMERIC");
                    writer.newLine();
                    writer.write("@attribute highSpectralCover" + 11000 + " NUMERIC");
                    writer.newLine();
            }
            for(int i=0;i<33;i++)
            {
                writer.write("@attribute mfcc"+i+" NUMERIC");
                writer.newLine();
            }
            for(int i=0;i<31;i++)
            {
                writer.write("@attribute delt1mfcc"+i+" NUMERIC");
                writer.newLine();
            }
            for(int i=0;i<29;i++)
            {
                writer.write("@attribute delt2mfcc"+i+" NUMERIC");
                writer.newLine();
            }
                writer.write("@attribute minPSD NUMERIC");
                writer.newLine();
                writer.write("@attribute cicada {yes,no}");
                writer.newLine();
                writer.write("@attribute rain {yes,no}");
                writer.newLine();
            writer.write("@data");
            writer.newLine();
            AcousticIndices.writeIndiciesSet(toFilter,writer,"yes","yes",hasHPF,isDownsampled);
            //AcousticIndices.writeIndiciesSet(toFilterCicada,writer,"no","yes",hasHPF,isDownsampled);
            /*AcousticIndices.writeIndiciesSet(toFilterRain,writer,"no","yes");
            AcousticIndices.writeIndiciesSet(toFilterRainCicada,writer,"yes","yes");*/
            //AcousticIndices.writeIndiciesSet(toFilterRainMedium,writer,"no","heavy");*/
            //AcousticIndices.writeIndiciesSet(toFilterRainHeavy,writer,"no","heavy");
            /*AcousticIndices.writeIndiciesSet(toFilterRainCicadaLight,writer,"yes","no");
            AcousticIndices.writeIndiciesSet(toFilterRainCicadaMedium,writer,"yes","heavy");
            AcousticIndices.writeIndiciesSet(toFilterRainCicadaHeavy,writer,"yes","heavy");*/
            //AcousticIndices.writeIndiciesSet(toFilterRainUncl,writer,"no","medium");
            writer.close();
        }
        catch(IOException e){e.printStackTrace();}
    }
	public static File[] getFilesInDirectoryWithExtension(String dir, String extension) {
        String filePath = dir;   //Path containing wav files
        File directory = new File(filePath);
        File[] files = directory.listFiles();
        List<File> wavFiles = new ArrayList<File>();
        if(files!=null) {
            for (int i = 0; i < files.length; i++) {
                String fileDir = files[i].getAbsolutePath();
                //System.out.println(fileDir.length()-fileDir.lastIndexOf(extension));
                if (fileDir.length() - fileDir.lastIndexOf(extension) == 4) {
                    wavFiles.add(files[i]);
                }
            }
            File[] wavArray = wavFiles.toArray(new File[0]);
            Arrays.sort(wavArray);
            return wavArray;
        }
        else
            return null;
    }
}
