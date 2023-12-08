import { HttpClient, HttpHeaders } from '@angular/common/http';
import { Component } from '@angular/core';
import { trigger, style, animate, transition } from '@angular/animations';

@Component({
  selector: 'app-home',
  templateUrl: './home.component.html',
  styleUrls: ['./home.component.scss'],
  animations: [
    trigger('scrollAnimation', [
      transition(':enter', [
        style({ opacity: 0, transform: 'translateY(-20px)' }),
        animate('300ms', style({ opacity: 1, transform: 'translateY(0)' })),
      ]),
    ]),
  ],
})
export class HomeComponent {
  showCompage = true;
  response:any;
  rawResponse:any;
  changeList:any = [];
  changeListFinalData:any = [];
  selectedIndex:any;
  showLoader = false;
  fileIcon:any = '';
  showFileMatchScore = false;

  file1: any;
  file2: any;
  constructor(
    private http: HttpClient,
  ){
  //  this.getData();
    this.file1 = null;
    this.file2 = null;
  }

  onFile1Selected(event: any) {
    this.file1 = event.target.files[0];
    if(this.file1.type == 'application/pdf'){
      this.fileIcon = '/assets/pdf.png';
    } else if(this.file1.type == 'application/vnd.ms-excel') {
      this.fileIcon = '/assets/xls.png';
    } else if(this.file1.type == 'application/vnd.openxmlformats-officedocument.presentationml.presentation') {
      this.fileIcon = '/assets/ppt.png';
    }
  }

  onFile2Selected(event: any) {
    this.file2 = event.target.files[0];
  }

  getData(){
    this.showLoader = true;
    const formData = new FormData();
    formData.append('file1', this.file1);
    formData.append('html_file', this.file2);

    // const headers = new HttpHeaders({
    //   'Content-Type': 'multipart/form-data',
    // });

    this.http.post('http://innovationhub.tvsnext.io:61077/process_and_compare_files',formData).subscribe((res:any)=>{
    this.showLoader = false;
    this.showCompage = true;
    console.log(res);
      this.rawResponse = JSON.parse(JSON.stringify(res));
      console.log(res.comparison_output.content);
      this.changeList = this.splitChangeContent(res.comparison_output.content);
      this.response = res;
     });
  }

  formateResponse(val:any){
    if(val){
      return val.replace(/\n/g, '<br>')
    } else {
      return "";
    }
     
  }

  splitChangeContent(res:any){
    let splitParentData = res.split("\n\n");
    if(splitParentData.length){
      splitParentData.forEach((element:any) => {
      if(this.splitSubContent(element)){
        this.changeListFinalData.push(this.splitSubContent(element));
      }
        
      });
    }
    return this.changeListFinalData;
  }

  splitSubContent(val:any){
    let data = val.split("\n");
    let retunObj:any = {};
    data.forEach((ele:any) =>{
       let objVal = ele.replace(/['"]+/g, '').split(":");
       console.log(objVal[0]);
       let key = objVal[0];
       let value = objVal[1];
       if(key && value){
        if(key=='Line Content'){
          retunObj.line_content = this.removeLeadingWhitespace(value);
        } else {
          retunObj[key] = value;
        }
       
       }
    
    })
  return retunObj;
  }


   highlightText(text:any) {
    // Get the input string and the target text
   let rawContent = this.rawResponse.pdf_text;

    // Check if the input string exists in the target text
    if (rawContent.includes(text)) {
      // Replace the target text with the highlighted version
      const highlightedText = rawContent.replace(
        new RegExp(text, 'g'),
        `<span class="highlight" id="highlightElement" [@scrollAnimation]>${text}</span>`

         
      );
      
      this.response.pdf_text = this.formateResponse(highlightedText);
      console.log(this.response.pdf_text);

      setTimeout(() => {
        this.scrollToElement();
      }, 20);

     

    } else {
     console.log('not there');
    }
  }


  removeLeadingWhitespace(inputString:any) {
    return inputString.replace(/^\s+/, '');
  }

  scrollToElement(): void {
    const element = document.querySelector(".highlight");
    if (element) {
      element.scrollIntoView({ behavior: 'smooth' });
    }
  }


  goBack(){
   this.showCompage = false;
   this.showFileMatchScore = false;
  }

}
