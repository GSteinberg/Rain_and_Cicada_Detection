import java.io.*;

/**
 * Created by aalle on 17/07/2017.
 */
public class ExternalProcessExecutor {
    public static void execute(String command)
    {
        try {
            System.out.println("Executing: "+command);
            Process p = Runtime.getRuntime().exec(command);
            InputStream err = p.getErrorStream();
            InputStreamReader errReader = new InputStreamReader(err);
            BufferedReader br = new BufferedReader(errReader);
            String line = null;
            System.out.println("<ERROR>");
            while ((line = br.readLine()) != null)
                System.out.println(line);
            System.out.println("</ERROR>");
            int exitVal = p.waitFor();
            p.destroy();
            System.out.println("Exit value: " + exitVal);
        } catch (IOException e) {
            e.printStackTrace();
        } catch (InterruptedException f) {
            f.printStackTrace();
        }
    }

    public static void moveTempFile(String baseName,String newName)
    {
        File oldFile=new File(baseName);
        File newFile=new File(newName);
        while(oldFile.delete())
        {
            System.out.println("Deleting old");
            try {
                Thread.sleep(50);
            }
            catch(InterruptedException e){e.printStackTrace();}
        }
        while(!newFile.renameTo(oldFile))
        {
            System.out.println("Renaming new "+oldFile.getName());
            try {
                Thread.sleep(50);
            }
            catch(InterruptedException e){e.printStackTrace();}
        }
    }
}
